# app/ingest_api.py
from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
import os, json, datetime, uuid, tempfile
import boto3

# adapt these imports to match your rag_utils
from app.rag_utils import load_text_file, split_docs, create_or_load_index  # existing helpers

router = APIRouter()
S3_BUCKET = os.environ.get("DATA_BUCKET", "")  # if empty -> fallback to local manifest
s3 = boto3.client("s3", region_name=os.environ.get("AWS_REGION", "us-east-1"))

class VersionIngestRequest(BaseModel):
    dataset_id: str
    source: str  # s3://... or local path for now
    version: str | None = None
    upsert: bool = False

def manifest_key(dataset_id: str):
    return f"datasets/{dataset_id}/manifest.json"

def _get_manifest_local(path):
    if os.path.exists(path):
        with open(path,"r") as f:
            return json.load(f)
    return {"dataset_id": os.path.basename(path), "versions": []}

def _put_manifest_local(path, manifest):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,"w") as f:
        json.dump(manifest, f, indent=2)

def get_manifest(dataset_id: str):
    if S3_BUCKET:
        try:
            obj = s3.get_object(Bucket=S3_BUCKET, Key=manifest_key(dataset_id))
            return json.loads(obj["Body"].read().decode())
        except s3.exceptions.NoSuchKey:
            return {"dataset_id": dataset_id, "versions": []}
        except Exception:
            return {"dataset_id": dataset_id, "versions": []}
    else:
        local_path = f"./data/{dataset_id}/manifest.json"
        return _get_manifest_local(local_path)

def put_manifest(dataset_id: str, manifest: dict):
    if S3_BUCKET:
        s3.put_object(Bucket=S3_BUCKET, Key=manifest_key(dataset_id),
                      Body=json.dumps(manifest).encode(), ContentType="application/json")
    else:
        local_path = f"./data/{dataset_id}/manifest.json"
        _put_manifest_local(local_path, manifest)

@router.post("/ingest/version")
async def ingest_version(req: VersionIngestRequest, background_tasks: BackgroundTasks):
    ds = req.dataset_id
    version = req.version or datetime.datetime.utcnow().strftime("v%Y%m%dT%H%M%S")
    manifest = get_manifest(ds)

    # duplicate handling
    if any(v["version"] == version for v in manifest["versions"]) and not req.upsert:
        raise HTTPException(status_code=400, detail="version exists; set upsert=true to overwrite")

    entry = {
        "version": version,
        "id": str(uuid.uuid4()),
        "created_at": datetime.datetime.utcnow().isoformat(),
        "source": req.source,
        "status": "PENDING",
        "chunks_indexed": 0,
        "index_path": None
    }

    # upsert: remove old entry and append new
    manifest["versions"] = [v for v in manifest["versions"] if v["version"] != version] + [entry]
    put_manifest(ds, manifest)

    background_tasks.add_task(_process_and_update_manifest, ds, version, req.source, entry["id"])

    return {"message": "ingest started", "dataset_id": ds, "version": version, "task_id": entry["id"]}

def _process_and_update_manifest(dataset_id, version, source, entry_id):
    # small wrapper to reuse your rag_utils pipeline (synchronous background)
    try:
        # 1) load docs (adapt path handling; support s3:// later)
        # If source starts with s3:// just raise for now (we'll add S3 support in next block)
        if source.startswith("s3://"):
            raise RuntimeError("S3 source not supported in local dev; upload file to sample_docs/ and use local path for now")

        docs = load_text_file(source)
        chunks = split_docs(docs)
        # store index locally under /tmp (use create_or_load_index to build)
        idx = create_or_load_index(chunks, persist_dir=f"/tmp/indexes/{dataset_id}/{version}")
        # update manifest
        manifest = get_manifest(dataset_id)
        for v in manifest["versions"]:
            if v["version"] == version and v["id"] == entry_id:
                v["status"] = "INDEXED"
                v["chunks_indexed"] = len(chunks)
                v["index_path"] = f"/tmp/indexes/{dataset_id}/{version}"
                v["indexed_at"] = datetime.datetime.utcnow().isoformat()
        put_manifest(dataset_id, manifest)
    except Exception as e:
        manifest = get_manifest(dataset_id)
        for v in manifest["versions"]:
            if v["version"] == version and v["id"] == entry_id:
                v["status"] = "FAILED"
                v["error"] = str(e)
        put_manifest(dataset_id, manifest)
        # don't re-raise so background task finishes gracefully

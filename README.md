This repository contains a Retrieval-Augmented Generation (RAG) microservice built with FastAPI, LangChain and deployed on AWS ECS Fargate behind an ALB.  
It uses AWS Secrets Manager for OpenAI keys and pushes Docker images to ECR.

See `.github/workflows/deploy.yml` for CI/CD pipeline.

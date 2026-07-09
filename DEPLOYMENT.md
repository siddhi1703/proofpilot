# Deploy ProofPilot to Alibaba Cloud

This project is packaged as one container: Vite builds the website, then
FastAPI serves both the API and the production frontend.

Official references:

- [Function Compute custom container images](https://www.alibabacloud.com/help/en/functioncompute/fc/user-guide/custom-container/)
- [Create a custom image function](https://www.alibabacloud.com/help/en/functioncompute/fc/create-a-custom-container-function-in-a-container-runtime)
- [Function Compute environment variables](https://www.alibabacloud.com/help/en/functioncompute/fc/user-guide/environment-variables)

## 1. Install Docker Desktop

Docker is required to build the image. On an Apple Silicon Mac, the image must
be built for `linux/amd64`, which is the architecture supported by Function
Compute.

## 2. Create Container Registry resources

In Alibaba Cloud Container Registry:

1. Select the same region you will use for Function Compute.
2. Create a Personal or supported Enterprise Edition instance.
3. Create a namespace, for example `proofpilot`.
4. Create an image repository, also called `proofpilot`.
5. Copy the registry login and repository commands shown by Alibaba Cloud.

Do not use ACR Economy Edition for Function Compute.

## 3. Build and push

Run these commands from the repository root. Replace every value in angle
brackets with the value shown in your ACR console.

```bash
docker login --username=<acr-username> <registry-host>

docker build \
  --platform linux/amd64 \
  -t <registry-host>/proofpilot/proofpilot:v1 .

docker push <registry-host>/proofpilot/proofpilot:v1
```

The Qwen key is not added during the image build.

## 4. Create the Function Compute function

In Function Compute:

1. Select the same region as the ACR repository.
2. Choose **Functions** → **Create Function**.
3. Select a **Custom Container** image from ACR.
4. Select the `proofpilot:v1` image.
5. Keep the image `CMD`; no custom start command is required.
6. Set the listening port (`CAPort`) to `9000`.
7. Set memory to at least `1024 MB`.
8. Set timeout to at least `180 seconds`.
9. Set instance concurrency to `1` for the hackathon demo.
10. Enable internet access so the function can call Qwen Cloud.

The server listens on `0.0.0.0:9000`, as Function Compute requires.

## 5. Add environment variables

On the function's **Configurations** → **Environment Variables** page, add:

```text
DASHSCOPE_API_KEY=<your real Qwen key>
QWEN_TEXT_MODEL=qwen3.6-flash
QWEN_EMBEDDING_MODEL=text-embedding-v4
QWEN_RERANK_MODEL=qwen3-rerank
```

Never place the real key in GitHub, the Dockerfile, or a screenshot.

## 6. Publish and verify

1. Deploy the function.
2. Wait until its state is **Active**.
3. Create or enable its public HTTP endpoint.
4. Open the endpoint; the ProofPilot website should appear.
5. Verify `<public-url>/health`.
6. Click **Load demo** and run one complete audit.

## 7. Updating the app

Build a new immutable tag such as `v2`, push it, then update the Function
Compute image selection. Avoid reusing `latest` during judging so the deployed
version remains reproducible.

# Deploying ProteinViz for free

## Streamlit Community Cloud (recommended, $0)

Community Cloud runs the app straight from this public GitHub repository.

1. Sign in at <https://share.streamlit.io> with your GitHub account.
2. Click **Create app**, then **Deploy a public app from GitHub**.
3. Fill in:
   - **Repository:** `shekibahmed/ProteinViz`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
   - **App URL:** pick a subdomain, e.g. `proteinviz` → `https://proteinviz.streamlit.app`
4. Open **Advanced settings** and choose **Python 3.12**. No secrets are needed.
5. Click **Deploy**. The first build takes a few minutes. Later pushes to `main` redeploy
   automatically.

Dependencies come from `requirements.txt` (which installs this repo with the `[app]` extra), and
settings from `.streamlit/config.toml`. Free apps go to sleep after a period without visitors and
wake on the next visit, which takes about 30 seconds.

After it is live, put the URL in the README badge and in the GitHub repo's **About → Website** field.

## Self-hosting with Docker

```bash
docker build -t proteinviz .
docker run -p 7860:7860 proteinviz      # http://localhost:7860
```

The image is torch-free, about 1 GB, and runs on any small CPU VM or a lab server.

> Hugging Face Docker Spaces now require a paid plan, so they are not used for the free demo.

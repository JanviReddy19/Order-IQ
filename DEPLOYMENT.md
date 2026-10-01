# Order IQ deployment checklist

## 1. GitHub

Push the repository with:

```bash
git add .
git commit -m "Order IQ real-data operations workflow"
git push origin main
```

## 2. Streamlit Community Cloud

1. Open `https://share.streamlit.io/`.
2. Connect the GitHub account.
3. Click **Create app**.
4. Choose the Order IQ repository.
5. Select branch `main`.
6. Set the entrypoint to `app.py`.
7. Deploy.

Streamlit's official deployment flow requires the repository, branch and entrypoint file; deployed apps receive a `streamlit.app` URL. See the official guide: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy

## 3. Real-data verification

Before publishing the project URL:

- confirm the sidebar is **not** in demo mode;
- confirm the banner says **REAL-DATA MODE**;
- confirm the data source is UCI Online Retail;
- record matched cancellation rate, median cancellation lag, top hotspot products, peak hour, top-decile precision/recall/lift and pilot-impact results;
- add 3–4 screenshots from the real-data app to `screenshots/`;
- replace the placeholder URL in `README.md`.

Never use demo-mode metrics in a resume.

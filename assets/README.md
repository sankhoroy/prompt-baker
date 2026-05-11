# Assets

- **`prompt-baker-logo.svg`** — default banner referenced from the root `README.md`. It is a normal file in the repo so GitHub can render it (relative path from `README.md`).

To use a **PNG** (or another name):

1. Add the file under `assets/` (for example `prompt-baker-logo.png`).
2. In the root `README.md`, set the image `src` to match, e.g. `assets/prompt-baker-logo.png`.
3. Run `git add assets/ README.md` and push.

If the image still does not show on GitHub, check that the file is on the **default branch** (usually `main`) and that the path and filename **match exactly** (case-sensitive on GitHub).

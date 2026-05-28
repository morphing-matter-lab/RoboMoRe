# Submission Checklist

Use this checklist before paper artifact submission.

## 1) Environment and reproducibility

- [ ] `requirements.txt` installs cleanly in a fresh environment.
- [ ] Main experiment command in `README.md` works with relative paths.
- [ ] Random seeds and core hyperparameters are documented.

## 2) Privacy and security

- [ ] No hardcoded API keys/tokens/passwords in `.py`, `.ipynb`, `.md`, `.json`, `.csv`.
- [ ] No personal absolute paths (for example `/home/<user>/...`).
- [ ] No private/proxy base URLs left in code or docs.
- [ ] Runtime secrets are provided only through environment variables.

## 3) Export sanitized package

From repository root:

```bash
python robomore/ant/different_llm_experiment/scripts/prepare_submission_package.py \
  --source . \
  --output ./submission_package \
  --force
```

## 4) Final quick scan (inside sanitized package)

```bash
rg -n "sk-[A-Za-z0-9_-]{12,}|API_KEY|TOKEN|SECRET|PASSWORD|/home/|/Users/|C:\\\\Users\\\\" ./submission_package
```

Expected: no real credentials or personal paths.

# Security Policy

## Reporting Security Issues

If you discover a security vulnerability in the Hybrid Assistant project, please **do not** open a public issue.

Instead, please email security concerns to the project maintainers with:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

We take all security reports seriously and will respond promptly.

## Security Considerations

### Local-Only Design
This project is designed to run locally on your machine:
- No cloud connectivity by default
- Camera and microphone access remain on your device
- Knowledge base stored locally in `learned_items/`

### Dependencies
- All Python dependencies are pinned in `requirements.txt`
- Keep dependencies updated: `pip install --upgrade -r requirements.txt`
- Review new dependencies before updating

### Camera & Privacy
- Camera access is requested explicitly by the user
- Video is processed locally; no frames are uploaded
- Learned knowledge and images stay on your device unless you explicitly share them

### Safe Practices
- Do not share `learned_items/` folder if it contains sensitive information
- Use CPU mode if GPU driver updates are untrusted
- Regularly backup your knowledge base (`learned_items/knowledge.json`)

## Supported Versions

| Version | Status | End of Life |
|---------|--------|------------|
| 1.0.0   | Active | 2026-11-17 |

Security patches will be provided for the active version.

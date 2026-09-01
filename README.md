# Privacy Vision Agent

Privacy Vision Agent is a Chrome browser extension and local server prototype for SIH problem statement 26171. It helps an agent operate browser interfaces from a screenshot and a natural-language task while keeping sensitive visual and form data on the device whenever possible: the extension detects faces and PII fields locally, redacts the screenshot, and sends only the sanitized context needed for action planning to a vision-language model.

## Architecture

The client is a Manifest V3 extension. It captures the visible tab, scans page controls in the content script, performs local face detection and PII redaction, and executes validated actions. The FastAPI server validates the request and asks the configured Groq vision-language model to choose an action.

```text
Screenshot + DOM scan
        |
        v
Local face detection and PII redaction
        |
        v
Compressed redacted image + selectors/labels + task text
        |
        v
FastAPI server -> Groq qwen/qwen3.6-27b VLM
        |
        v
Validated action (click/fill/scroll/wait/press)
        |
        v
Local browser execution
```

## Privacy design

The raw screenshot never leaves the device. It is used in the extension to draw the preview and create the redacted canvas, then the raw data URL is cleared. Detected faces are blurred locally, and recognized PII form-field regions are covered locally. The values entered into PII fields never leave the device; the server receives PII-field selectors and redaction reasons, not field values.

The server does receive the redacted, JPEG-compressed image, a bounded list of interactive element metadata (including selectors and visible labels where available), and the user's task text. Common email, phone, SSN, card, and credential-like patterns in the task are sanitized in the extension before transmission. Server logging is configured to avoid recording image data and API keys, but users should still treat task text and non-sensitive DOM metadata as network-bound request data.

## Setup

1. Clone the repository and enter the project directory:

   ```bash
   git clone <repository-url>
   cd privacy-vision-agent
   ```

2. Create and activate a Python virtual environment, then install the server dependencies:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. Create the environment file and add a Groq API key:

   ```bash
   cp .env.example .env
   ```

   Edit `.env` and set `GROQ_API_KEY` to your key. `GROQ_MODEL` is optional; the default is `qwen/qwen3.6-27b`.

4. Load the extension in Chrome:

   - Open `chrome://extensions`.
   - Enable **Developer mode**.
   - Select **Load unpacked** and choose the project root.

5. Start the local server from the project root with the virtual environment active:

   ```bash
   uvicorn server.main:app --host 127.0.0.1 --port 8000
   ```

   The extension expects the server at `http://127.0.0.1:8000`. The health endpoint is available at `/health`.

## Offline/local mode

Install [Ollama](https://ollama.com/download), then download the local vision model once:

```bash
ollama pull moondream
```

Start the server as usual, keep Ollama running, and select **Local (Ollama)** in the extension before running a task. The local provider keeps processing on-device with zero cloud calls. This directly fulfills the problem statement's allowance for any offline-deployable open-source or open-weights model. If Ollama is unavailable or the model has not been pulled, the server reports a 503 error instead of returning a fake action.

Moondream (~2GB) was chosen after testing showed larger vision models (qwen2.5vl:7b, llava-phi3) require more RAM than is available on typical modest hardware and get killed by the OS. This reflects a real resource/accuracy trade-off: smaller local models trade some accuracy for the ability to actually run on constrained devices — directly relevant to the problem statement's emphasis on lightweight, on-device-feasible processing.

## Usage

Open the extension popup on the page you want to inspect.

- **Capture Screen** captures the visible tab, locally blurs detected faces, locally covers detected PII fields, scans interactive controls, and shows the sanitized screenshot. It does not call the server.
- **Run Agent** requires a task in the input field. It performs the same local capture and redaction, sends the compressed redacted context to the server, receives a planned action, asks for confirmation for risky actions, and executes allowed actions in the active tab. It may capture the page again for a second planning round when the task is not complete.

The popup displays the latest face-model load, face-detection, DOM-scan, compressed-image, and server round-trip timings after capture or agent activity.

## Known limitations

- Task phrasing affects model selection and action planning; vague or unusual wording can produce no action or an incorrect target.
- Selectors and labels are generated from the current DOM and can be less reliable on complex, iframe-heavy, virtualized, or dynamically changing pages.
- Groq free-tier quotas and rate limits can affect availability and server latency.
- Firefox support is available through the separate `manifest.firefox.json`, but the Firefox workflow still requires manual browser testing.
- Local PII detection is heuristic and should not be treated as a guarantee that every sensitive value or visual detail is identified.

## Tech stack

- MediaPipe Tasks Vision local face detection using the CPU delegate by default. Both CPU and GPU (WebGL-backed) delegates were tested; CPU was chosen as the deliberate, evidence-based default because it measured meaningfully faster for this lightweight model on typical hardware, balancing latency against the potential compute savings of GPU acceleration. The GPU delegate remains available in the implementation comment for discrete-GPU hardware where it may perform better.
- Groq `qwen/qwen3.6-27b`, used by the server as the vision-language model.
- FastAPI and Uvicorn for the local HTTP server.
- Chrome Manifest V3 for the browser extension.
- Python, JavaScript, HTML, and CSS.

import { useCallback, useEffect, useRef, useState } from 'react'
import FieldTable from './components/FieldTable.jsx'
import ValidationPanel from './components/ValidationPanel.jsx'
import Dropzone from './components/Dropzone.jsx'

const API = '/api'

export default function App() {
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [busy, setBusy] = useState(false)
  const [health, setHealth] = useState(null)
  const [showRaw, setShowRaw] = useState(false)
  const abortRef = useRef(null)

  // Tell the user up front whether the backend is even reachable, rather than
  // letting them upload a file and wait for a confusing network error.
  useEffect(() => {
    fetch(`${API}/health`)
      .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`))))
      .then(setHealth)
      .catch(() => setHealth({ status: 'unreachable' }))
  }, [])

  // Object URLs are not garbage collected on their own.
  useEffect(() => {
    if (!file) return undefined
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const onSelect = useCallback((selected) => {
    setFile(selected)
    setResult(null)
    setError(null)
  }, [])

  const onExtract = useCallback(async () => {
    if (!file) return
    abortRef.current?.abort()
    const controller = new AbortController()
    abortRef.current = controller

    setBusy(true)
    setError(null)
    setResult(null)

    const body = new FormData()
    body.append('file', file)

    try {
      const response = await fetch(`${API}/extract`, {
        method: 'POST',
        body,
        signal: controller.signal,
      })
      const payload = await response.json().catch(() => null)
      if (!response.ok) {
        throw new Error(payload?.detail ?? `Request failed with HTTP ${response.status}`)
      }
      setResult(payload)
    } catch (err) {
      if (err.name !== 'AbortError') setError(err.message)
    } finally {
      setBusy(false)
    }
  }, [file])

  return (
    <div className="page">
      <header>
        <h1>Invoice extraction</h1>
        <p className="subtitle">
          Upload an invoice or receipt. A fine-tuned vision-language model reads it into
          structured JSON, then accounting rules check whether that reading is internally
          consistent.
        </p>
        <HealthBadge health={health} />
      </header>

      <main className="layout">
        <section className="panel">
          <h2>1 &middot; Document</h2>
          <Dropzone onSelect={onSelect} disabled={busy} />
          {previewUrl && (
            <figure className="preview">
              <img src={previewUrl} alt="Uploaded invoice preview" />
              <figcaption>{file?.name}</figcaption>
            </figure>
          )}
          <button className="primary" onClick={onExtract} disabled={!file || busy}>
            {busy ? 'Extracting…' : 'Extract fields'}
          </button>
          {busy && (
            <p className="hint">
              The first request loads the model onto the GPU and can take a minute.
            </p>
          )}
          {error && <p className="error">{error}</p>}
        </section>

        <section className="panel">
          <h2>2 &middot; Extracted fields</h2>
          {!result && <p className="empty">Nothing extracted yet.</p>}
          {result && !result.ok && (
            <div className="error-block">
              <p>{result.error}</p>
              <details>
                <summary>What the model actually returned</summary>
                <pre>{result.raw_output}</pre>
              </details>
            </div>
          )}
          {result?.ok && (
            <>
              <FieldTable extraction={result.extraction} />
              <p className="hint">Extracted in {result.elapsed_seconds}s</p>
              <button className="link" onClick={() => setShowRaw((value) => !value)}>
                {showRaw ? 'Hide' : 'Show'} raw JSON
              </button>
              {showRaw && <pre className="json">{JSON.stringify(result.extraction, null, 2)}</pre>}
            </>
          )}
        </section>

        <section className="panel">
          <h2>3 &middot; Validation</h2>
          {!result?.ok && <p className="empty">Nothing to validate yet.</p>}
          {result?.ok && <ValidationPanel validation={result.validation} />}
        </section>
      </main>

      <footer>
        <p>
          Trained on the public CORD-v2 receipt dataset plus synthetic French/Moroccan
          invoices. No real client data is used anywhere in this project.
        </p>
      </footer>
    </div>
  )
}

function HealthBadge({ health }) {
  if (!health) return <span className="badge badge-muted">checking API…</span>
  if (health.status !== 'ok') {
    return <span className="badge badge-error">API unreachable — is uvicorn running?</span>
  }
  return (
    <span className="badge badge-ok">
      API up{health.adapter ? ' · fine-tuned adapter' : ' · base model'}
    </span>
  )
}

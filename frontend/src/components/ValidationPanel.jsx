/**
 * The validation result is the part that makes the output usable.
 *
 * `severity: "ok"` means the arithmetic closes and the document could be posted
 * automatically. Anything else means a human should look at it, and the panel
 * says exactly which rule complained and why.
 */

const SEVERITY_COPY = {
  ok: {
    title: 'Consistent',
    body: 'Every applicable check passed. Safe to post automatically.',
  },
  warning: {
    title: 'Needs a look',
    body: 'The totals add up, but something else looks off.',
  },
  error: {
    title: 'Do not post',
    body: 'The document contradicts itself. A human has to review it.',
  },
}

const CHECK_LABELS = {
  required_fields: 'Required fields present',
  tax_arithmetic: 'Total HT + TVA = Total TTC',
  items_sum: 'Line items sum to Total HT',
  tax_rate_plausible: 'VAT rate is a standard Moroccan rate',
  date_valid: 'Date is real and plausible',
  ice_format: 'ICE is 15 digits',
  if_format: 'IF has a plausible length',
  currency_known: 'Currency is a known code',
}

export default function ValidationPanel({ validation }) {
  if (!validation) return null
  const severity = validation.severity ?? 'error'
  const copy = SEVERITY_COPY[severity] ?? SEVERITY_COPY.error

  const evaluated = validation.checks.filter((check) => check.status !== 'skipped')
  const skipped = validation.checks.filter((check) => check.status === 'skipped')

  return (
    <div className="validation">
      <div className={`verdict verdict-${severity}`}>
        <strong>{copy.title}</strong>
        <span>{copy.body}</span>
      </div>

      <ul className="checks">
        {evaluated.map((check) => (
          <li key={check.check} className={`check check-${check.status}`}>
            <span className="check-icon" aria-hidden="true">
              {check.status === 'pass' ? '✓' : '✕'}
            </span>
            <span className="check-body">
              <span className="check-name">{CHECK_LABELS[check.check] ?? check.check}</span>
              <span className="check-detail">{check.detail}</span>
            </span>
          </li>
        ))}
      </ul>

      {skipped.length > 0 && (
        <details className="skipped">
          <summary>
            {skipped.length} check{skipped.length === 1 ? '' : 's'} could not be run
          </summary>
          <p className="hint">
            These rules need fields this document does not carry. They are unverified, not
            failed &mdash; a receipt with no ICE is not a broken receipt.
          </p>
          <ul className="checks">
            {skipped.map((check) => (
              <li key={check.check} className="check check-skipped">
                <span className="check-icon" aria-hidden="true">
                  –
                </span>
                <span className="check-body">
                  <span className="check-name">{CHECK_LABELS[check.check] ?? check.check}</span>
                  <span className="check-detail">{check.detail}</span>
                </span>
              </li>
            ))}
          </ul>
        </details>
      )}
    </div>
  )
}

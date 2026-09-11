/** Renders the extracted schema as a form an accountant can scan. */

const SCALAR_FIELDS = [
  ['invoice_number', 'Invoice number'],
  ['date', 'Date'],
  ['ice', 'ICE'],
  ['if_number', 'IF'],
  ['subtotal', 'Total HT'],
  ['tax', 'TVA'],
  ['total', 'Total TTC'],
  ['currency', 'Currency'],
]

export default function FieldTable({ extraction }) {
  const items = Array.isArray(extraction?.items) ? extraction.items : []

  return (
    <div className="fields">
      <dl className="field-grid">
        {SCALAR_FIELDS.map(([key, label]) => (
          <div key={key} className="field">
            <dt>{label}</dt>
            <dd className={extraction?.[key] ? '' : 'missing'}>
              {extraction?.[key] || 'not found'}
            </dd>
          </div>
        ))}
      </dl>

      <h3>Line items</h3>
      {items.length === 0 ? (
        <p className="empty">No line items extracted.</p>
      ) : (
        <table className="items">
          <thead>
            <tr>
              <th>Description</th>
              <th className="num">Qty</th>
              <th className="num">Amount</th>
            </tr>
          </thead>
          <tbody>
            {items.map((item, index) => (
              <tr key={`${item?.name ?? 'item'}-${index}`}>
                <td>{item?.name || <span className="missing">—</span>}</td>
                <td className="num">{item?.qty || '—'}</td>
                <td className="num">{item?.price || '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

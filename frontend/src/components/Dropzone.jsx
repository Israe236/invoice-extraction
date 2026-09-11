import { useCallback, useRef, useState } from 'react'

const ACCEPTED = 'image/png,image/jpeg,image/webp,image/bmp,image/tiff'

/** Click-to-browse or drag-and-drop file picker for a single image. */
export default function Dropzone({ onSelect, disabled }) {
  const inputRef = useRef(null)
  const [dragging, setDragging] = useState(false)

  const handleFiles = useCallback(
    (files) => {
      const file = files?.[0]
      if (!file) return
      if (!file.type.startsWith('image/')) return
      onSelect(file)
    },
    [onSelect],
  )

  const onDrop = useCallback(
    (event) => {
      event.preventDefault()
      setDragging(false)
      if (disabled) return
      handleFiles(event.dataTransfer.files)
    },
    [disabled, handleFiles],
  )

  return (
    <div
      className={`dropzone${dragging ? ' dragging' : ''}${disabled ? ' disabled' : ''}`}
      onDragOver={(event) => {
        event.preventDefault()
        if (!disabled) setDragging(true)
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={onDrop}
      onClick={() => !disabled && inputRef.current?.click()}
      onKeyDown={(event) => {
        if (!disabled && (event.key === 'Enter' || event.key === ' ')) inputRef.current?.click()
      }}
      role="button"
      tabIndex={0}
      aria-label="Upload an invoice image"
    >
      <input
        ref={inputRef}
        type="file"
        accept={ACCEPTED}
        hidden
        disabled={disabled}
        onChange={(event) => handleFiles(event.target.files)}
      />
      <p className="dropzone-title">Drop an invoice image here</p>
      <p className="dropzone-hint">or click to browse &middot; PNG, JPEG, WebP &middot; up to 10 MB</p>
    </div>
  )
}

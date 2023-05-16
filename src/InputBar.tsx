import React, { useState, useRef, useEffect } from 'react'
import { TextField, InputAdornment, OutlinedInput } from '@mui/material'
import SendIcon from '@mui/icons-material/Send';

const InputBar = ({ hasMessages, disabled, onMessageSend, bottomRef }) => {
  const [message, setMessage] = useState('')
  const inputRef = useRef(null)

  const handleKeyPress = (event) => {
    if (event.key === 'Enter' && message.trim()) {
      onMessageSend(message)
      setMessage('')
    }
  }

  useEffect(() => {
    if (!disabled && inputRef.current) {
      inputRef.current.focus()
    }

  }, [disabled])

  const showAdornment = !disabled && message.trim().length > 0

  return (
    <div style={{ padding: '0.5rem' }}>
      <OutlinedInput
        ref={inputRef}
        autofocus
        fullWidth
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        onKeyPress={handleKeyPress}
        variant="outlined"
        disabled={disabled}
        size="large"
        placeholder="Tell me what you'd like to view"
        endAdornment={
          <InputAdornment position="end">
            <SendIcon style={{opacity: showAdornment ? 1 : 0.2}} />
          </InputAdornment>
        }
      />
      <div ref={bottomRef} />
    </div>
  )
}

export default InputBar

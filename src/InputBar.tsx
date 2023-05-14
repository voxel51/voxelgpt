import React, { useState } from 'react'
import { TextField } from '@mui/material'

const InputBar = ({ disabled, onMessageSend }) => {
  const [message, setMessage] = useState('')

  const handleKeyPress = (event) => {
    if (event.key === 'Enter' && message.trim()) {
      onMessageSend(message)
      setMessage('')
    }
  }

  return (
    <div style={{ padding: '0.5rem' }}>
      <TextField
        fullWidth
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        onKeyPress={handleKeyPress}
        variant="outlined"
        autofocus
        disabled={disabled}
        size="small"
        placeholder="Type your message"
      />
    </div>
  )
}

export default InputBar

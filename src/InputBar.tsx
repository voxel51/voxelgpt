import React, { useState, useRef, useEffect } from 'react'
import {
  TextField,
  InputAdornment,
  OutlinedInput,
  IconButton
} from '@mui/material'
import SendIcon from '@mui/icons-material/Send'
import { useRecoilState } from 'recoil'
import { atoms } from './state'

const InputBar = ({ hasMessages, disabled, onMessageSend, bottomRef }) => {
  const [message, setMessage] = useRecoilState(atoms.input)
  const inputRef = useRef(null)

  function sendMessage() {
    if (message.trim()) {
      onMessageSend(message)
      setMessage('')
    }
  }

  const handleKeyPress = (event) => {
    if (event.key === 'Enter') {
      sendMessage()
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
          <IconButton disabled={!showAdornment} onClick={sendMessage}>
            <SendIcon style={{ opacity: showAdornment ? 1 : 0.2 }} />
          </IconButton>
        }
      />
      <div ref={bottomRef} />
    </div>
  )
}

export default InputBar

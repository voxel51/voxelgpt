import React, { useState } from 'react';
import { TextField } from '@mui/material';

const InputBar = ({ onMessageSend }) => {
  const [message, setMessage] = useState('');

  const handleKeyPress = (event) => {
    if (event.key === 'Enter' && message.trim()) {
      onMessageSend(message);
      setMessage('');
    }
  };

  return (
    <div style={{background: '#fff', padding: '3px'}}>
      <TextField
        fullWidth
        value={message}
        onChange={(e) => setMessage(e.target.value)}
        onKeyPress={handleKeyPress}
        label="Type your message"
        variant="outlined"
        autofocus
      />
    </div>
  );
};

export default InputBar;
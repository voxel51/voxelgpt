import React, { useState, useRef, useEffect } from "react";
import {
  TextField,
  InputAdornment,
  OutlinedInput,
  IconButton,
} from "@mui/material";
import SendIcon from "@mui/icons-material/Send";
import { useRecoilState } from "recoil";
import { atoms } from "./state";

const InputBar = ({ hasMessages, disabled, onMessageSend, bottomRef }) => {
<<<<<<< HEAD
  const [waiting, setWaiting] = useRecoilState(atoms.waiting)
  const [message, setMessage] = useRecoilState(atoms.input)
  const inputRef = useRef(null)

  function sendMessage() {
    if (message.trim()) {
      setWaiting(true)
      onMessageSend(message)
      setMessage('')
=======
  const [message, setMessage] = useRecoilState(atoms.input);
  const inputRef = useRef(null);

  function sendMessage() {
    if (message.trim()) {
      onMessageSend(message);
      setMessage("");
>>>>>>> 09e000a (update intro view)
    }
  }

  const handleKeyPress = (event) => {
    if (event.key === "Enter") {
      sendMessage();
    }
  };

  useEffect(() => {
    if (!disabled && inputRef.current) {
      inputRef.current.focus();
    }
  }, [disabled]);

  const showAdornment = !disabled && message.trim().length > 0;

  return (
    <div style={{ padding: "0.5rem" }}>
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
        placeholder="Send a message..."
        endAdornment={
          <IconButton disabled={!showAdornment} onClick={sendMessage}>
            <SendIcon style={{ opacity: showAdornment ? 1 : 0.2 }} />
          </IconButton>
        }
      />
      <div ref={bottomRef} />
    </div>
  );
};

export default InputBar;

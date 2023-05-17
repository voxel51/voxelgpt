import React, { useEffect, useRef } from 'react'
import { useRecoilValue } from 'recoil'
import Message, { MessageWrapper } from './Message'
import * as state from './state'
import { ChatGPTAvatar } from './avatars'
import { Grid } from '@mui/material'

const Chat = () => {
  const bottomRef = useRef(null)
  const messages = useRecoilValue(state.atoms.messages)

  useEffect(() => {
    if (bottomRef.current && messages.length > 0) {
      bottomRef.current.scrollIntoView({ behavior: 'smooth' })
    }
  }, [messages])

  const avatars = {
    incoming: <ChatGPTAvatar />
  }

  return (
    <div style={{ overflow: 'auto' }}>
      <Grid container direction="row">
        {groupMessages(messages, avatars)}
      </Grid>
      <div ref={bottomRef} />
    </div>
  )
}

function groupMessages(messages, avatars) {
  const els = []
  let currentGroup = []
  let idx = 0
  for (const message of messages) {
    idx += 1
    // group messages by type
    if (currentGroup.length > 0 && currentGroup[0].type !== message.type) {
      console.log(message)
      const groupMessage = currentGroup[0]
      els.push(
        <MessageWrapper
          avatar={
            groupMessage.type === 'incoming'
              ? avatars.incoming
              : avatars.outgoing
          }
          type={groupMessage.type}
          index={idx}
          messages={currentGroup}
        />
      )
      currentGroup = [message]
    } else {
      currentGroup.push(message)
    }
  }
  if (currentGroup.length > 0) {
    const groupMessage = currentGroup[0]
    els.push(
      <MessageWrapper
        avatar={
          groupMessage.type === 'incoming' ? avatars.incoming : avatars.outgoing
        }
        type={groupMessage.type}
        index={idx}
        messages={currentGroup}
      />
    )
  }
  return els
}

export default Chat

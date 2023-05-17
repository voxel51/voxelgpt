import React from 'react'
import Avatar from '@mui/material/Avatar'
import Typography from '@mui/material/Typography'
import ReactMarkdown from 'react-markdown'
import { useTheme } from '@fiftyone/components'
import { Grid, Paper, Typography } from '@mui/material'
import useTypewriterEffect from './useTypewriterEffect'

export const Message = ({ type, avatar, content = '', button }) => {
  const animatedContent = useTypewriterEffect(
    type === 'incoming' ? content : '',
    5
  )

  return (
    <Typography component="div" style={{ marginLeft: 2 }}>
      <ReactMarkdown>
        {type === 'incoming'
          ? animatedContent.length === content.length
            ? content
            : animatedContent
          : content}
      </ReactMarkdown>
    </Typography>
  )
}

export function MessageWrapper({ index, type, avatar, messages }) {
  const theme = useTheme()
  const background =
    type === 'incoming' ? theme.background.header : theme.background.level1

  return (
    <Grid
      container
      sx={{ background, padding: '1rem' }}
      justifyContent="center"
    >
      <Grid container item lg={8} spacing={2}>
        <Grid item>{avatar || <Avatar alt={type} />}</Grid>
        <Grid item xs>
          {messages.map((message, index) => (
            <Message
              key={index}
              type={message.type}
              content={message.content}
              button={message.button}
            />
          ))}
        </Grid>
      </Grid>
    </Grid>
  )
}

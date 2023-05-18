import React from 'react'
import Avatar from '@mui/material/Avatar'
import ReactMarkdown from 'react-markdown'
import { useTheme } from '@fiftyone/components'
import { Grid, Box, Typography } from '@mui/material'
import {OperatorIO, types} from "@fiftyone/operators"
import LoadingIndicator from './LoadingIndicator'
import { ChatGPTAvatar } from './avatars'

export const Message = ({ type, avatar, content = '', outputs, data }) => {
  if (outputs)
    return (
      <OperatorIO
        schema={types.Property.fromJSON(outputs)}
        data={data}
        type="output"
      />
    )

  if (content)
    return (
      <Grid spacing={2} container sx={{ pl: 1 }}>
        <Grid item>
          <Typography component="p" m={"14px 0"}>
            {content}
          </Typography>
        </Grid>
      </Grid>
    )

  return null
}

export function MessageWrapper({ type, messages, receiving, last }) {
  const theme = useTheme()
  const isIncoming = type === 'incoming'
  const background =
    isIncoming ? theme.background.header : theme.background.level1

  return (
    <Grid
      container
      sx={{ background, padding: '1rem' }}
      justifyContent="center"
    >
      <Grid container item lg={8} spacing={2}>
        <Grid item>
          <Box mt={'5px'}>
            {isIncoming ? <ChatGPTAvatar /> : <Avatar alt="you" />}
          </Box>
        </Grid>
        <Grid container item xs>
          {messages.map((message, index) => (
            <Grid item>
              <Message
                key={index}
                type={message.type}
                {...message}
              />
            </Grid>
          ))}
          {receiving && (
            <Grid container item xs={12}>
              <Grid item>
                <LoadingIndicator />
              </Grid>
            </Grid>
          )}
        </Grid>
      </Grid>
    </Grid>
  )
}

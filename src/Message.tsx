import React from 'react'
import Avatar from '@mui/material/Avatar'
import ReactMarkdown from 'react-markdown'
import { useTheme } from '@fiftyone/components'
import { Grid, Box, Typography } from '@mui/material'
import {OperatorIO, types} from "@fiftyone/operators"
import LoadingIndicator from './LoadingIndicator'
import { ChatGPTAvatar } from './avatars'

export const Message = ({ type, avatar, content = '', outputs, data }) => {
  if (outputs) {
    const schema = types.Property.fromJSON(outputs)

    return (
      <OperatorIO
        schema={{
          ...schema,
          view: {
            ...schema.view,
            componentsProps: {
              gridContainer: {
                item: true,
                spacing: 0,
                sx: { pl: 0 }
              }
            }
          }
        }}
        data={data}
        type="output"
      />
    )
  }

  if (content)
    return (
      <Grid spacing={2} container>
        <Grid item style={{paddingLeft: '1rem'}}>
          <Typography component="p" my={1.5}>
            {content}
          </Typography>
        </Grid>
      </Grid>
    )

  return null
}

export function MessageWrapper({ type, messages, receiving, waiting, last }) {
  const theme = useTheme()
  const isIncoming = type === 'incoming'
  const background =
    isIncoming ? theme.background.header : theme.background.level1
  const showLoading = waiting || receiving;

  return (
    <Grid
      container
      sx={{ background, padding: '1rem', '& p': {m: 0, mt: 1} }}
      justifyContent="center"
    >
      <Grid container item lg={8} spacing={2} style={{minWidth: '500px'}}>
        <Grid item container xs={1}>
          <Grid item justifyContent="center">
            {isIncoming ? <ChatGPTAvatar /> : <Avatar alt="you" />}
          </Grid>
        </Grid>
        <Grid container item xs={11}>
          {messages.map((message, index) => (
            <Grid item xs={12} style={{paddingLeft: '1rem'}}>
              <Message
                key={index}
                type={message.type}
                {...message}
              />
            </Grid>
          ))}
          {showLoading && (
            <Grid container item xs={12} sx={{ pl: 1 }}>
              <Grid item>
                <Box my={1.5}>
                  <LoadingIndicator />
                </Box>
              </Grid>
            </Grid>
          )}
        </Grid>
      </Grid>
    </Grid>
  )
}

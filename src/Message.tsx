import React, { useEffect, useRef, useState } from 'react'
import Avatar from '@mui/material/Avatar'
import ReactMarkdown from 'react-markdown'
import { useTheme } from '@fiftyone/components'
import { Grid, Box, Typography, IconButton } from '@mui/material'
import {OperatorIO, types, executeOperator} from "@fiftyone/operators"
import LoadingIndicator from './LoadingIndicator'
import { ChatGPTAvatar } from './avatars'
import ThumbDown from '@mui/icons-material/ThumbDown'
import ThumbUp from '@mui/icons-material/ThumbUp'
import { useRecoilState } from 'recoil'
import {atoms} from './state'

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

// a hook given that returns state based on whether or not the mouse is hovered over the referenced element
function useHover() {
  const ref = useRef(null)
  const [hovered, setHovered] = React.useState(false)
  useEffect(() => {
    const onMouseOver = () => setHovered(true)
    const onMouseOut = () => setHovered(false)
    const elem = ref.current
    if (elem) {
      elem.addEventListener('mouseover', onMouseOver)
      elem.addEventListener('mouseout', onMouseOut)
      return () => {
        elem.removeEventListener('mouseover', onMouseOver)
        elem.removeEventListener('mouseout', onMouseOut)
      }
    }
  }, [ref.current])
  return {
    ref,
    hovered
  }
}

export function MessageWrapper({ type, messages, receiving, waiting, last }) {
  const theme = useTheme()
  const {ref, hovered} = useHover()
  const isIncoming = type === 'incoming'
  const background =
    isIncoming ? theme.background.header : theme.background.level1
  const showLoading = waiting || receiving;
  const queryId = messages[0]?.response_to;

  return (
    <Grid
      ref={ref}
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
        <Grid container item xs={9}>
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
        <Grid container item xs={2}>
          {isIncoming && <Vote queryId={queryId} hidden={!hovered} />}
        </Grid>
      </Grid>
    </Grid>
  )
}

function Vote({queryId, hidden}) {
  const [vote, setVote] = useRecoilState(atoms.votes(queryId))
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const hasVoted = vote && vote.direction;
  if (!queryId || hidden) return null;

  const showVoteUp = !hasVoted || vote.direction === 'upvote';
  const showVoteDown = !hasVoted || vote.direction === 'downvote';

  const handleVote = async (direction) => {
    let error = null;
    setIsLoading(true)
    try {
      await executeOperator("@voxel51/voxelgpt/vote_for_query", {query_id: queryId, vote: direction})
      setVote({direction})
    } catch (e) {
      console.error(e)
    }
    setIsLoading(false)
  }

  return (
    <div style={{marginTop: '3px', opacity: hasVoted ? 0.5 : 1}}>
      {showVoteUp && <IconButton disabled={hasVoted} onClick={() => handleVote('upvote')}>
        <ThumbUp style={{width: '18px'}} />
      </IconButton>}
      {showVoteDown && <IconButton disabled={hasVoted} onClick={() => handleVote('downvote')}>
        <ThumbDown style={{width: '18px'}} />
      </IconButton>}
    </div>
  )
}
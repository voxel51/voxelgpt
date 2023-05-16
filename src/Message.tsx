import React from 'react';
import Avatar from '@mui/material/Avatar';
import Typography from '@mui/material/Typography';
import ReactMarkdown from 'react-markdown';
import { useTheme } from '@mui/material/styles';
import {Button} from '@fiftyone/components'
import { Grid, Paper, Typography } from '@mui/material';
import useTypewriterEffect from './useTypewriterEffect';

export const Message = ({ type, avatar, content = '', button }) => {
  const theme = useTheme();
  const animatedContent = useTypewriterEffect(type === 'incoming' ? content : '', 5);


  return (
    <Typography component='div' style={{ marginLeft: theme.spacing(2) }}>
      <ReactMarkdown>
        {type === 'incoming' ? (animatedContent.length === content.length ? content : animatedContent) : content}
      </ReactMarkdown>
    </Typography>
  );
};


export function MessageWrapper({index,
  type,
  avatar,
  messages}) {
  const theme = useTheme();
  const background = type === 'incoming' ? "rgba(255, 255, 255, 0.04)" : "rgba(255, 255, 255, 0.08)";

  return (
    <Grid container sx={{background, padding: "1rem"}} justifyContent="center">
        <Grid container item lg={8} spacing={2}>
          <Grid item>
            {avatar || <Avatar alt={type} />}
          </Grid>
          <Grid item>
            {messages.map((message, index) => (
              <Message key={index} type={message.type} content={message.content} button={message.button} />
            ))}
          </Grid>
        </Grid>
    </Grid>
  )
}
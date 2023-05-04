import React from 'react';
import Avatar from '@mui/material/Avatar';
import Typography from '@mui/material/Typography';
import ReactMarkdown from 'react-markdown';
import { useTheme } from '@mui/material/styles';
import {Button} from '@fiftyone/components'
import { Grid, Paper, Typography } from '@mui/material';
import useTypewriterEffect from './useTypewriterEffect';

const Message = ({ type, avatar, content = '', button }) => {
  const theme = useTheme();
  const animatedContent = useTypewriterEffect(type === 'incoming' ? content : '', 50);


  if (button) {
    return (
      <div
        style={{
          display: 'flex',
          marginBottom: theme.spacing(2),
          justifyContent: 'center',
        }}
      >
        <Button>{button.label}</Button>
      </div>
    );
  }

  return (
    <div
      style={{
        display: 'flex',
        marginBottom: theme.spacing(2),
        justifyContent: 'center',
      }}
    >
      <Paper elevation={4} variant="outlined" style={{padding: '1rem', width: 500, display: 'flex', flexDirection: 'row',}}>
        <Avatar alt={type} src={avatar} />
        <Typography component='div' style={{ marginLeft: theme.spacing(2) }}>
          <ReactMarkdown>
            {type === 'incoming' ? (animatedContent.length === content.length ? content : animatedContent) : content}
          </ReactMarkdown>
        </Typography>
      </Paper>
    </div>
  );
};

export default Message;
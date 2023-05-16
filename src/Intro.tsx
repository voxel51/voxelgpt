import {
  Grid,
  Paper,
  Typography,
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  Avatar
} from '@mui/material';

import {EmojiObjects, QuestionAnswer, Dataset, Psychology} from "@mui/icons-material";

export const Intro = () => {
  return (
    <Grid container item direction="row" sx={{margin: 'auto'}} spacing={2} justifyContent="center" alignItems="start">
      <Grid item xs={12}>
        <Typography variant="h2" style={{textAlign: 'center'}}>
          VoxelGPT
        </Typography>
      </Grid>
      <Grid item sm={12} lg={4}>
        <Paper elevation={3} sx={{ padding: '20px', marginBottom: '16px' }}>
          <Typography variant="h4" gutterBottom style={{textAlign: 'center'}}>
            Examples
          </Typography>
          <List>
            <ListItem>
              <ListItemAvatar>
                <Avatar>
                  <QuestionAnswer />
                </Avatar>
              </ListItemAvatar>
              <ListItemText primary="Show me predicted airplanes" />
            </ListItem>
            <ListItem>
              <ListItemAvatar>
                <Avatar>
                  <QuestionAnswer />
                </Avatar>
              </ListItemAvatar>
              <ListItemText primary="Retrieve the first 10 images with 3 dogs and 1 cat" />
            </ListItem>
            <ListItem>
              <ListItemAvatar>
                <Avatar>
                  <QuestionAnswer />
                </Avatar>
              </ListItemAvatar>
              <ListItemText primary="How can I integrate GPT-3 into my application?" />
            </ListItem>
          </List>
        </Paper>
      </Grid>
      <Grid item sm={12} lg={4}>
        <Paper elevation={3} sx={{ padding: '20px', marginBottom: '16px' }}>
          <Typography variant="h4" gutterBottom style={{textAlign: 'center'}}>
            Capabilities
          </Typography>
          <List>
            <ListItem>
              <ListItemAvatar>
                <Avatar>
                  <EmojiObjects />
                </Avatar>
              </ListItemAvatar>
              <ListItemText primary="Understands the schema of your dataset" />
            </ListItem>
            <ListItem>
              <ListItemAvatar>
                <Avatar>
                  <Dataset />
                </Avatar>
              </ListItemAvatar>
              <ListItemText primary="Can run SQL-like queries on computer vision datasets" />
            </ListItem>
            <ListItem>
              <ListItemAvatar>
                <Avatar>
                  <Psychology />
                </Avatar>
              </ListItemAvatar>
              <ListItemText primary="Knows how to use brain methods, evaluations, similarity indexes, and more" />
            </ListItem>
          </List>
        </Paper>
      </Grid>
    </Grid>
  )
}
import {
  Avatar,
  Grid,
  List,
  ListItem,
  ListItemAvatar,
  ListItemButton,
  ListItemText,
  Paper,
  Typography
} from '@mui/material'

import Dataset from '@mui/icons-material/Dataset';
import EmojiObjects from '@mui/icons-material/EmojiObjects';
import Psychology from '@mui/icons-material/Psychology';
import QuestionAnswer from '@mui/icons-material/QuestionAnswer';


import { useSetRecoilState } from 'recoil'
import { atoms } from './state'

const examples = [
  { id: 'example-1', label: 'Show me predicted airplanes' },
  {
    id: 'example-2',
    label: 'Retrieve the first 10 images with 3 dogs and 1 cat'
  },
  { id: 'example-3', label: 'How can I integrate GPT-3 into my application?' }
]
const capabilities = [
  {
    id: 'capability-1',
    label: 'Understands the schema of your dataset',
    Icon: EmojiObjects
  },
  {
    id: 'capability-2',
    label: 'Can run SQL-like queries on computer vision datasets',
    Icon: Dataset
  },
  {
    id: 'capability-3',
    label:
      'Knows how to use brain methods, evaluations, similarity indexes, and more',
    Icon: Psychology
  }
]

export const Intro = () => {
  const setInput = useSetRecoilState(atoms.input)
  return (
    <Grid
      container
      item
      direction="row"
      sx={{ margin: 'auto' }}
      spacing={2}
      justifyContent="center"
      alignItems="start"
    >
      <Grid item xs={12}>
        <Typography variant="h2" style={{ textAlign: 'center' }}>
          VoxelGPT
        </Typography>
      </Grid>
      <Grid item sm={12} lg={4}>
        <Paper elevation={3} sx={{ padding: '20px', marginBottom: '16px' }}>
          <Typography variant="h4" gutterBottom style={{ textAlign: 'center' }}>
            Examples
          </Typography>
          <List>
            {examples.map(({ label, id }) => (
              <ListItemButton
                key={id}
                onClick={() => {
                  setInput(label)
                }}
              >
                <ListItemAvatar>
                  <Avatar>
                    <QuestionAnswer />
                  </Avatar>
                </ListItemAvatar>
                <ListItemText primary={label} />
              </ListItemButton>
            ))}
          </List>
        </Paper>
      </Grid>
      <Grid item sm={12} lg={4}>
        <Paper elevation={3} sx={{ padding: '20px', marginBottom: '16px' }}>
          <Typography variant="h4" gutterBottom style={{ textAlign: 'center' }}>
            Capabilities
          </Typography>
          <List>
            {capabilities.map(({ id, label, Icon }) => (
              <ListItem key={id}>
                <ListItemAvatar>
                  <Avatar>
                    <Icon />
                  </Avatar>
                </ListItemAvatar>
                <ListItemText primary={label} />
              </ListItem>
            ))}
          </List>
        </Paper>
      </Grid>
    </Grid>
  )
}

import {
  Avatar,
  Grid,
  List,
  ListItem,
  ListItemAvatar,
  ListItemButton,
  ListItemText,
  Paper,
  Typography,
} from "@mui/material";

import Dataset from "@mui/icons-material/Dataset";
import Psychology from "@mui/icons-material/Psychology";
import QuestionAnswer from "@mui/icons-material/QuestionAnswer";
import SchemaIcon from "@mui/icons-material/Schema";
import ManageSearchIcon from "@mui/icons-material/ManageSearch";

import { useSetRecoilState } from "recoil";
import { atoms } from "./state";

const examples = [
  { id: "example-1", label: "How do I export in COCO format?" },
  {
    id: "example-2",
    label: "What does the match() stage do?",
  },
  {
    id: "example-3",
    label:
      "Show me samples with a high confidence prediction evaluated as a false positive",
  },
  {
    id: "example-4",
    label: "Show me 10 images that contain dogs using text similarity",
  },
];
const capabilities = [
  {
    id: "capability-1",
    label: "Can search the FiftyOne docs for answers and link to its sources",
    Icon: ManageSearchIcon,
  },
  {
    id: "capability-2",
    label: "Understands the schema of your dataset",
    Icon: SchemaIcon,
  },
  {
    id: "capability-3",
    label: "Can automatically load views that contain the content you specify",
    Icon: Dataset,
  },
  {
    id: "capability-4",
    label:
      "Knows how to use brain methods, evaluations, similarity indexes, and more",
    Icon: Psychology,
  },
];

export const Intro = () => {
  const setInput = useSetRecoilState(atoms.input);
  return (
    <Grid
      container
      item
      direction="row"
      sx={{ margin: "auto" }}
      spacing={2}
      justifyContent="center"
      alignItems="start"
    >
      <Grid item xs={12}>
        <Typography variant="h2" style={{ textAlign: "center" }}>
          VoxelGPT
        </Typography>
      </Grid>
      <Grid item sm={12} lg={4} sx={{ alignSelf: "stretch", minWidth: 300 }}>
        <Paper
          elevation={3}
          sx={{ height: "100%", padding: "20px", marginBottom: "16px" }}
        >
          <Typography variant="h4" gutterBottom style={{ textAlign: "center" }}>
            Examples
          </Typography>
          <List>
            {examples.map(({ label, id }) => (
              <ListItemButton
                key={id}
                onClick={() => {
                  setInput(label);
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
      <Grid item sm={12} lg={4} sx={{ alignSelf: "stretch", minWidth: 300 }}>
        <Paper
          elevation={3}
          sx={{ height: "100%", padding: "20px", marginBottom: "16px" }}
        >
          <Typography variant="h4" gutterBottom style={{ textAlign: "center" }}>
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
  );
};

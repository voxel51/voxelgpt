import {
  Avatar,
  Grid,
  Icon,
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
import useContent from "./useContent";
import { useTheme } from "@fiftyone/components";


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
  const content = useContent();
  const theme = useTheme();
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
      <Grid
        container
        item
        direction="row"
        sx={{ margin: "auto" }}
        spacing={2}
        justifyContent="center"
        alignItems="center">
        <Grid item>
          {content.iconURL && (
            <img width={content.iconWidth + "px"} src={content.iconURL} alt="VoxelGPT" />
          )}
        </Grid>
        <Grid item>
          <Typography variant="h3" style={{ textAlign: "center", fontSize: "32px" }}>
            {content.mainHeaderLabel}
          </Typography>
        </Grid>
      </Grid>
      <Grid item sm={12} lg={4} sx={{ alignSelf: "stretch", minWidth: 300 }}>
        <CustomPaper
          header={"Example prompts:"}
          content={examples}
          theme={theme}
        />
      </Grid>
      <Grid item sm={12} lg={4} sx={{ alignSelf: "stretch", minWidth: 300 }}>
        <CustomPaper
          header={"Capabilities:"}
          content={capabilities}
          theme={theme}
        />
      </Grid>
    </Grid>
  );
};

function CustomPaper({header, content, theme}) {
  console.log(content)
  return (
    <Paper
      elevation={0}
      sx={{ height: "100%", padding: "20px", marginBottom: "16px", borderRadius: "15px", border: `solid 1px ${theme.divider}` }}
    >
      <Typography variant="h5" gutterBottom style={{ textAlign: "center" }}>
        {header}
      </Typography>
      <List>
        {content?.map(({ label, icon, Icon: IconCmpt }) => (
          <ListItemButton
            key={label}
            onClick={() => {
              setInput(label);
            }}
          >
            <ListItemAvatar>
              <Avatar sx={{borderRadius: '5px', background: theme.background.card, color: theme.text.secondary}}>
                {IconCmpt && <IconCmpt />}
                {icon ? (
                  <Icon>
                    {icon}
                  </Icon>
                ) : !IconCmpt && (
                  <QuestionAnswer />
                )}
              </Avatar>
            </ListItemAvatar>
            <ListItemText primary={label} />
          </ListItemButton>
        ))}
      </List>
    </Paper>
  )
}

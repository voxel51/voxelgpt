import {Grid, Button} from '@mui/material'
import * as state from "./state";
import {useRecoilValue, useResetRecoilState} from "recoil";
import Replay from '@mui/icons-material/Replay';
import StopCircle from '@mui/icons-material/StopCircle';
import {abortOperationsByURI} from "@fiftyone/operators"

const ASK_VOXELGPT_URI = '@voxel51/voxelgpt/ask_voxelgpt_panel'

export function Actions() {
  const receiving = useRecoilValue(state.atoms.receiving);
  const waiting = useRecoilValue(state.atoms.waiting);
  const resetReceiving = useResetRecoilState(state.atoms.receiving);
  const reset = useResetRecoilState(state.atoms.messages);
  const messages = useRecoilValue(state.atoms.messages);
  const handleStop = () => {
    resetReceiving();
    abortOperationsByURI(ASK_VOXELGPT_URI);
  }

  return (
    <Grid container justifyContent="center">
      {messages.length > 0 && !receiving && !waiting && <Grid item>
        <Button color="secondary" variant="contained" startIcon={<Replay />} onClick={() => reset()}>Start Over</Button>
      </Grid>}
      <Grid item>
        {receiving && <Button onClick={handleStop} color="secondary" variant="contained" startIcon={<StopCircle />}>Stop</Button>}
      </Grid> 
    </Grid>
  )
}
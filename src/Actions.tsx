import {Grid, Button} from '@mui/material'
import * as state from "./state";
import {useRecoilValue, useResetRecoilState} from "recoil";
import {Replay, StopCircle} from '@mui/icons-material';

export function Actions() {
  const receiving = useRecoilValue(state.atoms.receiving);
  const reset = useResetRecoilState(state.atoms.messages);
  const messages = useRecoilValue(state.atoms.messages);

  return (
    <Grid container justifyContent="center">
      {messages.length > 0 && !receiving && <Grid item>
        <Button color="secondary" variant="contained" startIcon={<Replay />} onClick={() => reset()}>Start Over</Button>
      </Grid>}
      <Grid item>
        {receiving && <Button color="secondary" variant="contained" startIcon={<StopCircle />}>Stop</Button>}
      </Grid> 
    </Grid>
  )
}
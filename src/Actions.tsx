import {Grid, Button} from '@mui/material'
import * as state from "./state";
import {useRecoilValue, useResetRecoilState} from "recoil";

export function Actions() {
  const receiving = useRecoilValue(state.atoms.receiving);
  const reset = useResetRecoilState(state.atoms.messages);
  const messages = useRecoilValue(state.atoms.messages);

  return (
    <Grid container>
      {messages.length > 0 && !receiving && <Grid item>
        <Button onClick={() => reset()}>Start Over</Button>
      </Grid>}
      <Grid item>
        {receiving && <Button>Stop</Button>}
      </Grid> 
    </Grid>
  )
}
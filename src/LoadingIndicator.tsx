import styled from 'styled-components';
import {Box} from '@mui/material';

const Animation = styled.div`
  font-weight: bold;
  font-family: monospace;
  font-size: 1.5rem;
  clip-path: inset(0 3ch 0 0);
  animation: l 1s steps(4) infinite;
  @keyframes l {
    to {
      clip-path: inset(0 -1ch 0 0)
    }
  }
  margin-top: -10px;
`

const Container = styled.div`
  border-radius: 3px;
  border: 1px solid #ccc;
  padding: 0.5rem;
  opacity: 0.5;
`
export default function LoadingIndicator() {
  return (
    <Container>
      <Animation>...</Animation>
    </Container>
  )
}
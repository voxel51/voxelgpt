import {registerOperator, Operator, OperatorConfig} from "@fiftyone/operators";
import * as state from "./state"
import {useRecoilState} from "recoil";

export class ShowMessage extends Operator {
  get config() {
    return new OperatorConfig({
      name: 'show_message',
      label: 'Show Message',
    })
  }

  useHooks() {
    const [messages, setMessages] = useRecoilState(state.atoms.messages)
    return {
      addMessage: (message) => {
        setMessages(current => [...current, message])
      }
    }
  }

  async execute(ctx) {
    ctx.hooks.addMessage({
      type: 'incoming',
      content: ctx.params.message
    })
  }
}


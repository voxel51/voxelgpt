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
      },
      updateLastIncomingMessage: (message) => {
        setMessages(current => {
          const lastIncomingMessage = current.filter(m => m.type === 'incoming').pop()
          if (lastIncomingMessage) {
            return [
              ...current.filter(m => m !== lastIncomingMessage),
              {
                type: 'incoming',
                ...lastIncomingMessage,
                ...message
              }
            ]
          }
          return current
        })
      }
    }
  }

  async execute(ctx) {
    if (ctx.params.message || ctx.params.outputs) {
      ctx.state.set(state.atoms.receiving, true)
      ctx.state.set(state.atoms.waiting, false)
      const {overwrite_last} = ctx.params.data || {}
      if (overwrite_last) {
        ctx.hooks.updateLastIncomingMessage(ctx.params)
      } else {
        ctx.hooks.addMessage({
          type: 'incoming',
          ...ctx.params
        })
      }
    }
    if (ctx.params.done) {
      ctx.state.set(state.atoms.receiving, false)
      ctx.state.set(state.atoms.waiting, false)
    }
  }
}


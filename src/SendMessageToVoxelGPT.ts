import {
  registerOperator,
  Operator,
  OperatorConfig,
  types,
  executeOperator,
} from "@fiftyone/operators";
import * as state from "./state";
import { useRecoilState } from "recoil";
import { uuid } from "./utils";
import { GPTMessage, GPTMessageType } from "./types";

export class SendMessageToVoxelGPT extends Operator {
  get config() {
    return new OperatorConfig({
      name: "send_message_to_voxelgpt",
      label: "Send Message to VoxelGPT",
    });
  }

  useHooks() {
    const [messages, setMessages] = useRecoilState(state.atoms.messages);
    return {
      messages,
      addMessage: (message) => {
        setMessages((current) => [...current, message]);
      },
    };
  }

  async execute(ctx) {
    const message = new GPTMessage(GPTMessageType.SUCCESS, [
      new types.Property(new types.String(), {
        default: ctx.params.message,
        readOnly: true,
      }),
    ]);
    ctx.hooks.addMessage({
      type: "outgoing",
      content: ctx.params.message,
    });
    await executeOperator(`${this.pluginName}/ask_voxelgpt_panel`, {
      query: ctx.params.message,
      history: ctx.hooks.messages,
    });
  }
}

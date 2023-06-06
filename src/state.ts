import { atom, atomFamily, selector } from "recoil";
import { getBrowserStorageEffectForKey } from "@fiftyone/state";

const PLUGIN_NAME = "@voxel51/voxelgpt";

export const atoms = {
  messages: atom({
    key: "messages",
    default: [],
    effects: [
      getBrowserStorageEffectForKey(`${PLUGIN_NAME}/messages`, {
        prependDatasetNameInKey: true,
        useJsonSerialization: true,
      }),
    ],
  }),
  receiving: atom({
    key: "receiving",
    default: false,
  }),
  waiting: atom({
    key: "waiting",
    default: false,
  }),
  input: atom({
    key: "voxel-gpt-input",
    default: "",
  }),
  votes: atomFamily({
    key: "voxel-gpt-votes",
    default: {},
  })
};



export const selectors = {};

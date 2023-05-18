import { atom, selector } from 'recoil'
import {getBrowserStorageEffectForKey} from "@fiftyone/state";

const PLUGIN_NAME = '@voxel51/fiftyone-gpt'

export const atoms = {
  messages: atom({
    key: 'messages',
    default: [],
    effects: [
      getBrowserStorageEffectForKey(`${PLUGIN_NAME}/messages`, {
        prependDatasetNameInKey: true,
        useJsonSerialization: true
      })
    ]
  }),
  receiving: atom({
    key: 'receiving',
    default: false
  }),
  input: atom({
    key: 'voxel-gpt-input',
    default: ''
  })
}
export const selectors = {}

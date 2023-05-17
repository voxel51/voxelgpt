import { atom, selector } from 'recoil'

export const atoms = {
  messages: atom({
    key: 'messages',
    default: []
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

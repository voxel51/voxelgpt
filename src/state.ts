import { atom, selector } from 'recoil'

export const atoms = {
  messages: atom({
    key: 'messages',
    default: [],
  }),
  receiving: atom({
    key: 'receiving',
    default: false,
  }),
}
export const selectors = {}
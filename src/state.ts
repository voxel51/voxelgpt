import { atom, selector } from 'recoil'

export const atoms = {
  messages: atom({
    key: 'messages',
    default: [],
  })
}
export const selectors = {}
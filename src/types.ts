import {types} from '@fiftyone/operators'
import { uuid } from './utils'

export enum GPTMessageType {
  SUCCESS = 'success',
  ERROR = 'error',
}

export class GPTMessage {
  public id: string = uuid()
  constructor(
    public type: string,
    public content: types.Property[]
  ) {}
}
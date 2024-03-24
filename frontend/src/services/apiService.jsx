// *****************************************************************************
// llm-gateway - A proxy service in front of llm models to encourage the
// responsible use of AI.

// Copyright 2023 Wealthsimple Technologies

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//   http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// *****************************************************************************

import axios from 'axios';
import { modelChoices } from '../constants';

import {
  Role
} from "./../app/interfaces";


export async function fetchResponseFromModel(
  messageHistory,
) {
  const requestBody = {"text": messageHistory.filter(a => a.role === Role.user).map(a => a.content).join("\n")};
  const url = "http://localhost:5000/chat";
  try {
    const res = await axios.post(url, requestBody);
    return res.data.response;
  } catch (error) {
    const err = error;
    const errRes = err.response;

    if (err.message && errRes.data.detail) {
      throw new Error(`${err.message}: ${errRes.data.detail}`);
    }

    throw new Error('Something went wrong - check console for details.');
  }
}

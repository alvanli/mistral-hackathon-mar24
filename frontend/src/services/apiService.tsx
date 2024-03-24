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

import axios, { type AxiosError, type AxiosResponse } from 'axios';
import { type IRequestBody } from '../app/interfaces';
import { modelChoices } from '../constants';

export async function fetchResponseFromModel(
  props: IRequestBody,
): Promise<string> {
  const requestBody = modelChoices[props.model].requestBody(props);
  const url = modelChoices[props.model].apiEndpoint;
  try {
    const res: AxiosResponse = await axios.post(url, requestBody);
    return modelChoices[props.model].responseHandler(res.data);
  } catch (error) {
    const err = error as AxiosError;
    const errRes = err.response as AxiosResponse;

    if (err.message && errRes.data.detail) {
      throw new Error(`${err.message}: ${errRes.data.detail}`);
    }

    throw new Error('Something went wrong - check console for details.');
  }
}

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

import React, { useState } from 'react';
import { ThemeToggleButton } from '../ThemeProvider';
import { DeleteIcon } from '../Icons/TrashCanIcon';
import { ClearChatConfirmDialogComponent } from './ClearChatConfirmDialog';

import Tabs from '@mui/material/Tabs';
import Tab from '@mui/material/Tab';

interface ModelBannerProps {
  modelName: string;
  isModelLoadingReply: boolean;
  currentTab: number;
  handleTabChange: (event: React.SyntheticEvent, newValue: number) => void;
  clearCurrentChatMessagesAction: () => void;
}

export const ModelBannerComponent: React.FC<ModelBannerProps> = (
  props: ModelBannerProps,
) => {
  const [showClearChatConfirmDialog, setShowClearChatConfirmDialog] =
    useState<boolean>(false);

  return (
    <>
      <article
        className={`model-banner ${
          props.isModelLoadingReply ? 'cursor-disable' : ''
        }`}
      >
        <Tabs value={props.currentTab} onChange={props.handleTabChange} sx={{ width: 300 }}>
          <Tab label="🗨️" id='simple-tab-0' sx={{ width: 100 }} />
          <Tab label="🗺️" id='simple-tab-1' sx={{ width: 100 }} />
        </Tabs>
        <div className="banner-actions">
          <div className="theme-toggle-button">
            <ThemeToggleButton />
          </div>
          <div
            className="clear-chat"
            onClick={() => setShowClearChatConfirmDialog(true)}
          >
            {DeleteIcon}
          </div>
        </div>
      </article>
      <ClearChatConfirmDialogComponent
        showDialog={showClearChatConfirmDialog}
        setShowDialog={setShowClearChatConfirmDialog}
        clearCurrentChatAction={props.clearCurrentChatMessagesAction}
      />
    </>
  );
};

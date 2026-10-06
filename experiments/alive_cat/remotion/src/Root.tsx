import React from 'react';
import {Composition} from 'remotion';
import {Scene} from './Scene';
export const Root: React.FC = () => (
  <Composition id="Scene" component={Scene} durationInFrames={180} fps={30} width={1920} height={1080} />
);

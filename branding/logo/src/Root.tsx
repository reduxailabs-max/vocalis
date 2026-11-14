import React from 'react';
import {Composition, Still} from 'remotion';
import {Logo} from './Logo';

// Two stills: the app icon (with a dark tile) and the bare mark (transparent).
export const Root: React.FC = () => {
  return (
    <>
      <Still id="Logo" component={() => <Logo tile />} width={1024} height={1024} />
      <Still id="LogoMark" component={() => <Logo tile={false} />} width={1024} height={1024} />
      <Composition
        id="LogoPulse"
        component={() => <Logo tile />}
        durationInFrames={1}
        fps={30}
        width={1024}
        height={1024}
      />
    </>
  );
};

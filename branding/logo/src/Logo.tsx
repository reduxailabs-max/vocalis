import React from 'react';

const RED = '#e53935';
const RED_BRIGHT = '#ff1744';
const RED_DEEP = '#8e0e14';
const BLACK = '#0d0d0f';
const BLACK_LIFT = '#1b1b20';

/** Wave strokes: mirrored on both sides, tallest next to the rings. */
const WAVE_BARS = [
  {offset: 335, height: 200},
  {offset: 375, height: 140},
  {offset: 415, height: 100},
  {offset: 455, height: 60},
];

export const Logo: React.FC<{tile: boolean}> = ({tile}) => {
  const cx = 512;
  const cy = 512;
  return (
    <svg width="1024" height="1024" viewBox="0 0 1024 1024" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="tile" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor={BLACK_LIFT} />
          <stop offset="1" stopColor={BLACK} />
        </linearGradient>
        <linearGradient id="mic" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor={RED_BRIGHT} />
          <stop offset="1" stopColor={RED} />
        </linearGradient>
        <linearGradient id="micLine" gradientUnits="userSpaceOnUse" x1="0" y1="300" x2="0" y2="720">
          <stop offset="0" stopColor={RED_BRIGHT} />
          <stop offset="1" stopColor={RED} />
        </linearGradient>
        <radialGradient id="halo" cx="0.5" cy="0.5" r="0.5">
          <stop offset="0" stopColor={RED} stopOpacity="0.55" />
          <stop offset="0.6" stopColor={RED_DEEP} stopOpacity="0.25" />
          <stop offset="1" stopColor={RED_DEEP} stopOpacity="0" />
        </radialGradient>
        <filter id="glow" x="-30%" y="-30%" width="160%" height="160%">
          <feGaussianBlur stdDeviation="9" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {tile ? <rect x="24" y="24" width="976" height="976" rx="220" fill="url(#tile)" /> : null}
      {tile ? (
        <rect x="24" y="24" width="976" height="976" rx="220" fill="none" stroke={RED_DEEP} strokeWidth="6" />
      ) : null}

      <circle cx={cx} cy={cy} r="330" fill="url(#halo)" />

      {/* glowing rings */}
      <g fill="none" filter="url(#glow)">
        <circle cx={cx} cy={cy} r="300" stroke={RED} strokeOpacity="0.35" strokeWidth="6" />
        <circle cx={cx} cy={cy} r="262" stroke={RED} strokeOpacity="0.6" strokeWidth="8" />
        <circle cx={cx} cy={cy} r="224" stroke={RED_BRIGHT} strokeOpacity="0.9" strokeWidth="10" />
      </g>

      {/* sound waves on both sides */}
      <g stroke="url(#micLine)" strokeWidth="24" strokeLinecap="round" filter="url(#glow)">
        {WAVE_BARS.map(({offset, height}) => (
          <React.Fragment key={offset}>
            <line x1={cx - offset} y1={cy - height / 2} x2={cx - offset} y2={cy + height / 2} />
            <line x1={cx + offset} y1={cy - height / 2} x2={cx + offset} y2={cy + height / 2} />
          </React.Fragment>
        ))}
      </g>

      {/* microphone */}
      <g filter="url(#glow)">
        <rect x="448" y="330" width="128" height="240" rx="64" fill="url(#mic)" />
        <path
          d="M 392 500 A 120 120 0 0 0 632 500"
          fill="none"
          stroke="url(#micLine)"
          strokeWidth="24"
          strokeLinecap="round"
        />
        <line x1="512" y1="620" x2="512" y2="690" stroke="url(#micLine)" strokeWidth="24" strokeLinecap="round" />
        <line x1="450" y1="690" x2="574" y2="690" stroke="url(#micLine)" strokeWidth="24" strokeLinecap="round" />
      </g>
    </svg>
  );
};

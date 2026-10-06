import React from 'react';
import {AbsoluteFill, Img, staticFile, useCurrentFrame, useVideoConfig, interpolate, spring, Easing} from 'remotion';

const font = `@font-face{font-family:Caveat;src:url(${staticFile('Caveat-Variable.ttf')});}`;
// координаты исходника 1264x848; кот (86,175)-(461,488)
const S = 1920 / 1264, OY = (1080 - 848 * S) / 2;
const CAT = {x: 86, y: 175, w: 375, h: 313};

export const Scene: React.FC = () => {
  const f = useCurrentFrame();
  const {fps} = useVideoConfig();
  // камера: медленный наезд к центру между котом и письмом
  const z = interpolate(f, [0, 180], [1, 1.08], {easing: Easing.inOut(Easing.cubic)});
  // присед 0.4-0.8 c, прыжок пружиной с 0.8 c
  const squash = interpolate(f, [12, 24, 26, 30], [0, 1, 1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const jump = spring({frame: f - 24, fps, config: {damping: 9, stiffness: 120}});
  const up = Math.sin(Math.min(Math.max((f - 24) / 14, 0), 1) * Math.PI) * 70;
  const landing = interpolate(f, [38, 41, 47], [0, 1, 0], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  // дыхание после приземления
  const breath = f > 47 ? Math.sin((f - 47) / fps * Math.PI * 2 / 2.4) * 0.018 : 0;
  // поворот головы не делаем (один слой), но наклон всего тела к письму
  const lean = interpolate(f, [70, 95], [0, 4], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.inOut(Easing.quad)});
  const sx = 1 + 0.06 * squash - 0.04 * landing * -1 - breath * 0.5;
  const sy = 1 - 0.09 * squash - 0.07 * landing + breath;
  // тень сжимается, когда кот в воздухе
  const shadow = 1 - up / 140;
  // слово от руки: прорисовка 2.6–3.6 c
  const write = interpolate(f, [78, 108], [0, 100], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.out(Easing.quad)});
  void jump;
  return (
    <AbsoluteFill style={{backgroundColor: '#FAF7F0', overflow: 'hidden'}}>
      <style>{font}</style>
      <AbsoluteFill style={{transform: `scale(${z})`, transformOrigin: '42% 55%'}}>
        <div style={{position: 'absolute', left: 0, top: OY, width: 1264 * S, height: 848 * S}}>
          <Img src={staticFile('bg.png')} style={{width: '100%', height: '100%'}} />
          <div style={{position: 'absolute', left: (CAT.x + 60) * S, top: (CAT.y + CAT.h - 14) * S,
            width: 230 * S, height: 22 * S, borderRadius: '50%', background: 'rgba(60,55,50,0.10)',
            transform: `scaleX(${shadow})`, filter: 'blur(6px)'}} />
          <Img src={staticFile('cat.png')} style={{position: 'absolute', left: CAT.x * S, top: CAT.y * S,
            width: CAT.w * S, height: CAT.h * S,
            transformOrigin: '55% 100%',
            transform: `translateY(${-up * S}px) rotate(${lean}deg) scale(${sx}, ${sy})`}} />
          <div style={{position: 'absolute', left: 560 * S, top: 92 * S, fontFamily: 'Caveat', fontWeight: 700,
            fontSize: 92 * S, color: '#2b2a28', clipPath: `inset(0 ${100 - write}% 0 0)`}}>не сейчас…</div>
        </div>
      </AbsoluteFill>
    </AbsoluteFill>
  );
};

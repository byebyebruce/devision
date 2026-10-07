/* Hand-drawn diagrams are inputs only. Predictions always come from the API. */
(() => {
  const diagram = (background, drawing) => 'data:image/svg+xml;charset=utf-8,' + encodeURIComponent(
    `<svg xmlns="http://www.w3.org/2000/svg" width="640" height="420" viewBox="0 0 640 420"><rect width="640" height="420" fill="${background}"/>${drawing}</svg>`
  );
  const yesno = (text) => ({ type: 'noul', text });
  const choice = (text, options) => ({ type: 'choice', text, options });
  const photos = {
    '1-dog-skier.jpg': {
      id: 'snow', title: '雪地里的伙伴', subtitle: '物体识别 · 环境 · 动作',
      description: '从狗、雪地和滑雪者开始，试试一张照片里的不同问题。',
      questions: [yesno('Is there a dog in the image?'), choice('What is covering the ground?', ['snow', 'grass', 'sand']), choice('What is the person doing?', ['skiing', 'cycling', 'swimming'])],
    },
    '2-tennis-lying.jpg': {
      id: 'tennis', title: '球场休息时刻', subtitle: '人数 · 运动 · 物体',
      description: '结合人物和球拍判断运动，也可以修改选项，观察答案概率怎样变化。',
      questions: [choice('How many people are in the image?', ['one', 'two', 'three', 'four']), choice('Which sport is associated with the equipment?', ['tennis', 'baseball', 'soccer']), yesno('Is anyone wearing sunglasses?')],
    },
  };
  const diagrams = [
    {
      id: 'count', kind: 'diagram', title: '颜色与数量', subtitle: '合成示例 · 属性 · 计数',
      description: '三个珊瑚红的圆。试试问数量、颜色和形状，也可以改选项看概率怎样变化。',
      url: diagram('#f4e8dc', '<circle cx="145" cy="210" r="56" fill="#cc7358"/><circle cx="320" cy="210" r="56" fill="#cc7358"/><circle cx="495" cy="210" r="56" fill="#cc7358"/>'),
      questions: [choice('How many circles are in the image?', ['two', 'three', 'four', 'five']), choice('What color are the circles?', ['red', 'green', 'blue', 'yellow']), yesno('Are the circles red?')],
    },
    {
      id: 'spatial', kind: 'diagram', title: '换个方向看', subtitle: '合成示例 · 空间关系',
      description: '左右关系仍是模型的挑战。打开水平镜像对照，观察图片翻转后答案是否随之改变。',
      url: diagram('#e9eee6', '<path d="M90 324H550" stroke="#c9d5c1" stroke-width="2"/><circle cx="204" cy="217" r="73" fill="#cc7158"/><rect x="350" y="143" width="146" height="146" rx="6" fill="#6f8da8"/>'),
      questions: [choice('Is the red circle to the left or to the right of the blue square?', ['left', 'right']), yesno('Is the blue square to the right of the red circle?')],
    },
    {
      id: 'size', kind: 'diagram', title: '大小之间', subtitle: '合成示例 · 比较 · 形状',
      description: '相同形状，不同颜色与大小。改写问题中的目标物体，探索模型是否选中了你指的对象。',
      url: diagram('#eceaf1', '<circle cx="215" cy="210" r="99" fill="#b3a1c8"/><circle cx="458" cy="240" r="51" fill="#d1ab65"/>'),
      questions: [choice('Which circle is larger?', ['the purple circle', 'the yellow circle']), yesno('Is the yellow circle smaller than the purple circle?'), choice('How many circles are there?', ['one', 'two', 'three'])],
    },
    {
      id: 'science', kind: 'diagram', title: '磁极小实验', subtitle: '合成示例 · 科学探索',
      description: '两根条形磁铁，相邻端为 S 与 N。科学示意图是探索场景；可以加一个无图对照，看看图片带来多少变化。',
      url: diagram('#f0ebde', '<rect x="55" y="155" width="220" height="110" rx="10" fill="#cc7058"/><path d="M165 155h100q10 0 10 10v90q0 10-10 10H165Z" fill="#718f9e"/><rect x="365" y="155" width="220" height="110" rx="10" fill="#cc7058"/><path d="M475 155h100q10 0 10 10v90q0 10-10 10H475Z" fill="#718f9e"/><g fill="white" font-family="Arial,sans-serif" font-size="45" font-weight="bold" text-anchor="middle"><text x="110" y="226">N</text><text x="220" y="226">S</text><text x="420" y="226">N</text><text x="530" y="226">S</text></g><path d="M295 210h50" stroke="#b3aa94" stroke-width="3" stroke-dasharray="6 7"/>'),
      questions: [choice('Will these two magnets attract or repel each other?', ['attract', 'repel']), yesno('Are opposite magnetic poles facing each other?')],
    },
  ];
  window.devisionScenes = {
    diagrams,
    fromFile(name) {
      return { ...(photos[name] || { id: `file:${name}`, title: name, subtitle: '自定义图片 · 自由提问', description: '为这张图片编辑英文问题和候选答案。', questions: [yesno('Is there a person in the image?')] }), kind: 'photo', url: `/examples/${encodeURIComponent(name)}` };
    },
  };
})();

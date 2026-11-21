/* skrrt docs: copy buttons, the phone menu, and the statusline ruler. No dependencies. */
(function () {
  'use strict';
  var d = document;

  /* copy-to-clipboard on code blocks */
  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text);
    }
    return new Promise(function (resolve, reject) {
      var ta = d.createElement('textarea');
      ta.value = text;
      ta.setAttribute('readonly', '');
      ta.style.position = 'fixed';
      ta.style.top = '-100px';
      d.body.appendChild(ta);
      ta.select();
      try { d.execCommand('copy') ? resolve() : reject(); } catch (e) { reject(e); }
      d.body.removeChild(ta);
    });
  }
  Array.prototype.forEach.call(d.querySelectorAll('figure.code'), function (fig) {
    var bar = fig.querySelector('.code-bar');
    var code = fig.querySelector('code');
    if (!bar || !code || fig.hasAttribute('data-nocopy')) { return; }
    var btn = d.createElement('button');
    btn.type = 'button';
    btn.className = 'copy';
    btn.textContent = 'copy';
    btn.setAttribute('aria-label', 'Copy this code to the clipboard');
    var timer;
    btn.addEventListener('click', function () {
      var text = code.textContent.replace(/\n+$/, '');
      copyText(text).then(function () { flash('copied', true); }, function () { flash('failed', false); });
    });
    function flash(label, ok) {
      btn.textContent = label;
      btn.classList.toggle('done', ok);
      clearTimeout(timer);
      timer = setTimeout(function () { btn.textContent = 'copy'; btn.classList.remove('done'); }, 1600);
    }
    bar.appendChild(btn);
  });

  /* phone menu: a <details>, so it works without this; these are niceties */
  var menu = d.querySelector('details.menu');
  if (menu) {
    d.addEventListener('keydown', function (e) {
      if (e.key === 'Escape' && menu.open) { menu.open = false; menu.querySelector('summary').focus(); }
    });
    d.addEventListener('click', function (e) {
      if (menu.open && !menu.contains(e.target)) { menu.open = false; }
    });
  }

  /* statusline ruler: Top / 42% / Bot, like Vim's */
  var ruler = d.getElementById('ruler');
  if (ruler) {
    var ticking = false;
    var update = function () {
      ticking = false;
      var el = d.documentElement;
      var max = el.scrollHeight - window.innerHeight;
      var y = window.pageYOffset || el.scrollTop;
      var label;
      if (max <= 4) { label = 'All'; }
      else if (y <= 2) { label = 'Top'; }
      else if (y >= max - 2) { label = 'Bot'; }
      else { label = Math.round(y / max * 100) + '%'; }
      ruler.textContent = label;
    };
    window.addEventListener('scroll', function () {
      if (!ticking) { ticking = true; window.requestAnimationFrame(update); }
    }, { passive: true });
    window.addEventListener('resize', update);
    update();
  }
})();

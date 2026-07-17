// "Crear libro" page (templates/index.html): upload flow, live book-writing
// area, and the finish modal.

$(function () {
  const $form = $('#upload-form');
  const $fileInput = $('#file-input');
  const $uploadBtn = $('#upload-btn');
  const $progressArea = $('#progress-area');
  const $progressBar = $('#progress-bar');
  const $progressLabel = $('#progress-label');
  const $progressPercent = $('#progress-percent');
  const $resultMsg = $('#result-msg');
  const $servicesList = $('#services-list');
  const $errorArea = $('#error-area');
  const $errorMsg = $('#error-msg');
  const $generateBtn = $('#generate-btn');
  const $engineError = $('#engine-error');
  const $engineErrorMsg = $('#engine-error-msg');
  const $engineRetryBtn = $('#engine-retry-btn');

  // Fallback only — normally the server sends the wording, so it stays in one
  // place. This is for when our own server is what's unreachable.
  const ENGINE_DOWN_TEXT =
    'El motor de escritura no está disponible en este momento. ' +
    'Inténtalo de nuevo en unos minutos.';
  const $bookArea = $('#section-step-3');
  const $bookContent = $('#book-content');
  const $tocList = $('#book-toc-list');
  const $bookProgress = $('#book-progress');
  const $bookProgressLabel = $('#book-progress-label');
  const $bookProgressBar = $('#book-progress-bar');
  const $bookModal = $('#book-modal');
  const $modalDownloadBtn = $('#modal-download-btn');
  const $modalAddBtn = $('#modal-add-btn');
  const $modalAddMsg = $('#modal-add-msg');
  const $modalGotoHomeBtn = $('#modal-goto-home-btn');
  const $modalNewAdventureBtn = $('#modal-new-adventure-btn');

  let currentSessionId = null;

  // Switches the nav item and the matching <section> to "active" (styled/
  // shown in index.less), so only one step is visible at a time.
  function setStep(n) {
    $('nav > ol > li').removeClass('active');
    $('#step-' + n).addClass('active');
    $('section').removeClass('active');
    $('#section-step-' + n).addClass('active');
  }

  $form.on('submit', function (e) {
    e.preventDefault();

    const file = $fileInput[0].files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    // Reset UI
    $errorArea.hide();
    $progressArea.show();
    $uploadBtn.prop('disabled', true);
    $progressBar.val(0);
    $progressPercent.text('0%');
    $progressLabel.text('Subiendo...');

    $.ajax({
      url: '/upload',
      type: 'POST',
      data: formData,
      processData: false,
      contentType: false,
      xhr: function () {
        const xhr = new XMLHttpRequest();
        xhr.upload.addEventListener('progress', function (e) {
          if (e.lengthComputable) {
            const pct = Math.round((e.loaded / e.total) * 100);
            $progressBar.val(pct);
            $progressPercent.text(pct + '%');
            if (pct === 100) {
              $progressLabel.text('Procesando archivo...');
            }
          }
        });
        return xhr;
      },
      success: function (data) {
        $uploadBtn.prop('disabled', false);
        $progressArea.hide();
        renderUploadResult(data);
      },
      error: function (xhr) {
        $uploadBtn.prop('disabled', false);
        $progressArea.hide();
        let errText = 'Error del servidor.';
        try {
          const errData = JSON.parse(xhr.responseText);
          errText = errData.error || errText;
        } catch (_) {}
        showError(errText);
      },
    });
  });

  function showError(msg) {
    $errorMsg.text(msg);
    $errorArea.show();
  }

  // --- writing engine availability -----------------------------------------
  // The LLM runs on a separate box that isn't always up, and writes one book
  // at a time — so it can also be busy with someone else's. Rather than let the
  // reader start a ten-minute generation that dies on step 4, ask first — and
  // say so plainly if a run dies anyway. The server picks the wording for each
  // case; this banner just shows it, with a button to ask again.

  // Both of these leave the generate button alone: the banner reports what we
  // knew a moment ago, and "busy" or "down" both stop being true on their own.
  // Disabling the button would make the reader's obvious move — press it
  // again — do nothing at all, with no hint that the retry button is the only
  // live one. Pressing generate re-checks and either starts or says why not.
  function showEngineError(msg) {
    $engineErrorMsg.text(msg || ENGINE_DOWN_TEXT);
    $engineError.show();
  }

  function hideEngineError() {
    $engineError.hide();
  }

  // Resolves to true/false — never rejects, so a caller can just branch on it.
  function checkEngine() {
    return $.get('/engine/status')
      .then(function (data) {
        if (data.available) {
          hideEngineError();
          return true;
        }
        showEngineError(data.message);
        return false;
      })
      .catch(function () {
        // Our own server is unreachable, not the engine. Same dead end for the
        // reader, so same message.
        showEngineError(ENGINE_DOWN_TEXT);
        return false;
      });
  }

  $engineRetryBtn.on('click', function () {
    $engineRetryBtn.prop('disabled', true);
    checkEngine().always(function () {
      $engineRetryBtn.prop('disabled', false);
    });
  });

  function renderUploadResult(data) {
    currentSessionId = data.session_id || currentSessionId;

    const names = Object.keys(data.services);
    $resultMsg.text(
      'Archivo procesado. Se encontraron ' + names.length +
      ' servicios con un total de ' + data.total_items + ' elementos.'
    );

    // Previously-disabled services (e.g. when resuming a session) start unchecked.
    const disabledSet = new Set(data.disabled_services || []);

    let html = '<h3>Servicios encontrados:</h3><p><small>Desmarca los que no quieras incluir en tu libro.</small></p><ul id="services-checklist">';
    for (const [name, count] of Object.entries(data.services)) {
      const checked = disabledSet.has(name) ? '' : 'checked';
      html += '<li><label><input type="checkbox" class="service-checkbox" value="' +
        escapeHtml(name) + '" ' + checked + '> ' +
        escapeHtml(name) + ' — ' + count + ' elementos</label></li>';
    }
    html += '</ul>';

    if (data.skipped_services && data.skipped_services.length > 0) {
      html += '<h3>Servicios omitidos:</h3><ul>';
      for (const name of data.skipped_services) {
        html += '<li>' + escapeHtml(name) + '</li>';
      }
      html += '</ul>';
    }

    html += '<p><small>ID de sesión: ' + escapeHtml(currentSessionId) + '</small></p>';
    $servicesList.html(html);

    setStep(2);
  }

  // --- Book writing area -----------------------------------------------
  // Streams the LLM output into #book-content as it arrives (system-only,
  // read-only "shared doc" feel). Markdown-style "#" lines become headings
  // and are mirrored as clickable links in the #book-toc-list index.
  //
  // Generation runs server-side in a background thread tied to a cookie
  // (no login), so it keeps writing even if this tab is closed. Reopening
  // the page reconnects to /generate/stream, which replays everything
  // written so far before continuing live.

  $generateBtn.on('click', function () {
    if (!currentSessionId) return;

    const disabledServices = [];
    $('#services-list .service-checkbox').each(function () {
      if (!this.checked) disabledServices.push(this.value);
    });

    // Only for the length of the check — a second click would fire a second
    // check, not a second book, but the button shouldn't look live while we're
    // mid-question either.
    $generateBtn.prop('disabled', true);

    // Ask the engine before committing: a run that starts against a dead or
    // taken box burns the session (there's no resume) and shows the reader a
    // failure several minutes in, instead of right now.
    checkEngine().then(function (ok) {
      if (!ok) {
        // showEngineError already said why. Hand the button back so pressing
        // it again re-asks — by then the other book may well have finished.
        $generateBtn.prop('disabled', false);
        return;
      }
      startGeneration(disabledServices);
    });
  });

  function startGeneration(disabledServices) {
    $.ajax({
      url: '/generate/configure',
      type: 'POST',
      contentType: 'application/json',
      data: JSON.stringify({ disabled_services: disabledServices }),
    })
      .done(function () {
        setStep(3);
        $bookContent.empty();
        $tocList.empty();
        $bookArea[0].scrollIntoView({ behavior: 'smooth', block: 'start' });
        streamBook();
      })
      .fail(function (xhr) {
        $generateBtn.prop('disabled', false);
        let errText = 'Error del servidor.';
        let errKind = null;
        try {
          const errData = JSON.parse(xhr.responseText);
          errText = errData.error || errText;
          errKind = errData.error_kind || null;
        } catch (_) {}
        // Someone else claimed the engine between our check and this call.
        // Same dead end as an engine that's down, so use the same banner.
        if (errKind === 'busy') {
          showEngineError(errText);
        } else {
          showError(errText);
        }
      });
  }

  // On load, check whether this browser (via cookie) already has a
  // session — resume showing/streaming the book if one is in progress.
  $.get('/generate/status')
    .done(function (data) {
      if (!data.has_session) return;

      renderUploadResult(data);

      if (data.book_status && data.book_status !== 'none') {
        $generateBtn.prop('disabled', true);
        $('#services-list .service-checkbox').prop('disabled', true);
        setStep(3);
        $bookContent.empty();
        $tocList.empty();
        streamBook();
      }
    })
    .fail(function () { /* no previous session — normal first visit */ });

  // Progress banner shown before and between chapters. Both phases count, so
  // the bar fills: step-by-step while planning, chapter-by-chapter while
  // writing. It only goes indeterminate before the first event arrives.
  function showProgress(p) {
    $bookProgress.show();
    $bookProgressLabel.text(p.label || 'Preparando…');
    const done = p.phase === 'prep' ? p.step : p.chapter;
    if (p.total && done > 0) {
      $bookProgressBar.attr('max', p.total).attr('value', done);
    } else {
      $bookProgressBar.removeAttr('value'); // indeterminate
    }
  }

  function hideProgress() {
    $bookProgress.hide();
  }

  // --- Following the text as it's written --------------------------------
  // #book-content is its own scroll box (max-height + overflow-y in
  // index.less), so the prose scrolls inside it and the page stays put.
  //
  // The rule: stick to the bottom while the reader is at the bottom, and get
  // out of the way the moment they scroll off to read something earlier —
  // yanking them back mid-sentence is the thing that makes live logs unusable.
  // Returning to the bottom themselves resumes it.

  const bookContentEl = $bookContent[0];
  // Sub-pixel rounding and zoom levels mean "at the bottom" is never exactly 0,
  // and the last line is usually mid-render. A few px of slack absorbs both.
  const FOLLOW_SLACK_PX = 40;
  let following = true;

  function atBottom() {
    return bookContentEl.scrollHeight - bookContentEl.scrollTop -
      bookContentEl.clientHeight <= FOLLOW_SLACK_PX;
  }

  // Derived from where the box actually is, rather than tracked as an intent.
  // That's what makes it symmetric — and it's why followTail() below doesn't
  // have to suppress this handler: its own scroll lands at the bottom, which
  // re-arms following, which is exactly right.
  $bookContent.on('scroll', function () {
    following = atBottom();
  });

  function followTail() {
    if (following) {
      // Instant, not smooth: a smooth scroll per token never finishes before
      // the next one restarts it, and the text ends up permanently lagging.
      bookContentEl.scrollTop = bookContentEl.scrollHeight;
    }
  }

  function streamBook() {
    let lineBuffer = '';
    let currentParagraph = null;
    let liveLineEl = null;
    let headerCount = 0;
    let gotAnything = false;

    // A stream always starts by following. On a reconnect the server replays
    // the whole book so far in one go, and the tail is where the reader left
    // off — the same place a fresh run starts from.
    following = true;

    // Show something immediately so the page never looks stuck while the
    // first server event (outline planning) is on its way.
    showProgress({ label: 'Conectando…' });

    const source = new EventSource('/generate/stream');

    source.addEventListener('message', function (e) {
      let payload;
      try {
        payload = JSON.parse(e.data);
      } catch (err) {
        return;
      }
      gotAnything = true;
      if (payload.error) {
        hideProgress();
        if (payload.error_kind === 'engine') {
          // Not "something went wrong" — a specific, temporary thing the
          // reader can wait out. Send them back to the button that starts it.
          setStep(2);
          showEngineError(payload.error);
        } else {
          showError('Error generando el libro: ' + payload.error);
        }
        return;
      }
      if (payload.progress) {
        showProgress(payload.progress);
      }
      if (payload.content) {
        ingestChunk(payload.content);
      }
    });

    source.addEventListener('done', function () {
      source.close();
      hideProgress();
      $generateBtn.prop('disabled', false);
      checkAndShowModalIfDone();
    });

    source.addEventListener('error', function () {
      source.close();
      hideProgress();
      $generateBtn.prop('disabled', false);
      // Dying before a single event means the stream never opened — the server
      // refused us (someone beat us to the engine) rather than dropped us
      // mid-book. EventSource can't show us the reason, so go ask for it, and
      // put the reader back on the button they'd need to press.
      if (!gotAnything) {
        setStep(2);
        checkEngine();
      }
    });

    function ingestChunk(text) {
      lineBuffer += text;
      let idx;
      while ((idx = lineBuffer.indexOf('\n')) !== -1) {
        const line = lineBuffer.slice(0, idx);
        lineBuffer = lineBuffer.slice(idx + 1);
        finalizeLine(line);
      }
      updateLiveLine(lineBuffer);
      followTail();
    }

    function finalizeLine(line) {
      if (liveLineEl) {
        liveLineEl.remove();
        liveLineEl = null;
      }

      const trimmed = line.trim();
      const headingMatch = trimmed.match(/^(#{1,6})\s*(.+)$/);

      if (headingMatch) {
        const title = headingMatch[2].trim();
        if (title) {
          headerCount += 1;
          const id = 'chapter-' + headerCount;

          const $heading = $('<h3>').attr('id', id).text(title);
          $bookContent.append($heading);

          const $link = $('<a>').attr('href', '#' + id).text(title);
          $link.on('click', function (ev) {
            ev.preventDefault();
            $heading[0].scrollIntoView({ behavior: 'smooth', block: 'start' });
          });
          const $li = $('<li>').append($link);
          $tocList.append($li);
        }
        currentParagraph = null;
      } else if (trimmed === '') {
        currentParagraph = null;
      } else {
        if (!currentParagraph) {
          currentParagraph = $('<p>')[0];
          $bookContent.append(currentParagraph);
        }
        currentParagraph.textContent =
          (currentParagraph.textContent ? currentParagraph.textContent + ' ' : '') + line;
      }
    }

    function updateLiveLine(text) {
      if (!text) {
        if (liveLineEl) {
          liveLineEl.remove();
          liveLineEl = null;
        }
        return;
      }
      if (!liveLineEl) {
        liveLineEl = $('<p>').addClass('live-line')[0];
        $bookContent.append(liveLineEl);
      }
      liveLineEl.textContent = text;
    }
  }

  // --- Finish modal ------------------------------------------------------
  // Shown once generation is confirmed "done" (not just stopped/errored).

  function checkAndShowModalIfDone() {
    $.get('/generate/status').done(function (data) {
      if (data.has_session && data.book_status === 'done') {
        $bookModal.show();
      }
    });
  }

  $modalDownloadBtn.on('click', function () {
    window.location.href = '/book/download';
  });

  $modalAddBtn.on('click', function () {
    $modalAddBtn.prop('disabled', true);
    $modalAddMsg.hide();

    $.post('/library/add')
      .done(function (data) {
        $modalAddMsg.show();
        if (data.ok) {
          $modalAddMsg.text('Añadido a la colección.');
          $modalGotoHomeBtn.show();
        } else {
          $modalAddMsg.text(data.error || 'Error al añadir a la colección.');
          $modalAddBtn.prop('disabled', false);
        }
      })
      .fail(function () {
        $modalAddMsg.show();
        $modalAddMsg.text('Error de red.');
        $modalAddBtn.prop('disabled', false);
      });
  });

  $modalGotoHomeBtn.on('click', function () {
    window.location.href = '/';
  });

  // Resets the cookie-bound session and clears the story so the user can
  // start over. Anything already added to the library is left alone.
  $modalNewAdventureBtn.on('click', function () {
    const confirmed = window.confirm(
      'Esto borrará la historia actual (a menos que ya la hayas añadido a tu colección). ¿Quieres crear una nueva aventura?'
    );
    if (!confirmed) return;

    $modalNewAdventureBtn.prop('disabled', true);

    $.post('/session/reset')
      .done(function () {
        window.location.href = '/crear';
      })
      .fail(function () {
        $modalNewAdventureBtn.prop('disabled', false);
        alert('No se pudo reiniciar la sesión. Inténtalo de nuevo.');
      });
  });
});

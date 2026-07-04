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
  const $resultArea = $('#result-area');
  const $resultMsg = $('#result-msg');
  const $servicesList = $('#services-list');
  const $errorArea = $('#error-area');
  const $errorMsg = $('#error-msg');
  const $generateBtn = $('#generate-btn');
  const $bookArea = $('#book-area');
  const $bookContent = $('#book-content');
  const $tocList = $('#book-toc-list');
  const $bookModal = $('#book-modal');
  const $modalDownloadBtn = $('#modal-download-btn');
  const $modalAddBtn = $('#modal-add-btn');
  const $modalAddMsg = $('#modal-add-msg');
  const $modalGotoHomeBtn = $('#modal-goto-home-btn');

  let currentSessionId = null;

  $form.on('submit', function (e) {
    e.preventDefault();

    const file = $fileInput[0].files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    // Reset UI
    $errorArea.hide();
    $resultArea.hide();
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

  function renderUploadResult(data) {
    $resultArea.show();

    const names = Object.keys(data.services);
    $resultMsg.text(
      'Archivo procesado. Se encontraron ' + names.length +
      ' servicios con un total de ' + data.total_items + ' elementos.'
    );

    let html = '<h3>Servicios encontrados:</h3><ul>';
    for (const [name, count] of Object.entries(data.services)) {
      html += '<li>' + escapeHtml(name) + ' — ' + count + ' elementos</li>';
    }
    html += '</ul>';

    if (data.skipped_services && data.skipped_services.length > 0) {
      html += '<h3>Servicios omitidos:</h3><ul>';
      for (const name of data.skipped_services) {
        html += '<li>' + escapeHtml(name) + '</li>';
      }
      html += '</ul>';
    }

    html += '<p><small>ID de sesión: ' + escapeHtml(data.session_id) + '</small></p>';
    $servicesList.html(html);

    currentSessionId = data.session_id;
    $generateBtn.show();
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

    $generateBtn.prop('disabled', true);
    $bookArea.show();
    $bookContent.empty();
    $tocList.empty();
    $bookArea[0].scrollIntoView({ behavior: 'smooth', block: 'start' });

    streamBook();
  });

  // On load, check whether this browser (via cookie) already has a
  // session — resume showing/streaming the book if one is in progress.
  $.get('/generate/status')
    .done(function (data) {
      if (!data.has_session) return;

      renderUploadResult(data);

      if (data.book_status && data.book_status !== 'none') {
        $generateBtn.prop('disabled', true);
        $bookArea.show();
        $bookContent.empty();
        $tocList.empty();
        streamBook();
      }
    })
    .fail(function () { /* no previous session — normal first visit */ });

  function streamBook() {
    let lineBuffer = '';
    let currentParagraph = null;
    let liveLineEl = null;
    let headerCount = 0;

    const source = new EventSource('/generate/stream');

    source.addEventListener('message', function (e) {
      let payload;
      try {
        payload = JSON.parse(e.data);
      } catch (err) {
        return;
      }
      if (payload.error) {
        showError('Error generando el libro: ' + payload.error);
        return;
      }
      if (payload.content) {
        ingestChunk(payload.content);
      }
    });

    source.addEventListener('done', function () {
      source.close();
      $generateBtn.prop('disabled', false);
      checkAndShowModalIfDone();
    });

    source.addEventListener('error', function () {
      source.close();
      $generateBtn.prop('disabled', false);
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
});

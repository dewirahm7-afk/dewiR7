// frontend/js/app.js - VERSION LENGKAP dengan Tab 2 Translate
// --- GLOBAL util + state ---
const $ = (id) => document.getElementById(id);

// hindari [object Object], simpan URL string & peta tempo segmen
if (!window._exPreview) window._exPreview = { url: null, tempoMap: {} };

class DracinApp {
    constructor() {
        this.currentSessionId = null;
        this.isConnected = false;
        this.baseUrl = '';
        this.activeTab = 'srt-processing';
        this.ocr = null;
        // Add these lines
        this.videoFile = null;
        this.srtFile = null;
        this.exMap = { global_nudge_ms: 0, row_nudges: {} };
		this.exportBusy = false;
        this.initializeApp();
    }

    initializeApp() {
        console.log('Initializing DracinDub Web App...');
        
        // Set up tab navigation
        this.setupTabNavigation();
        
        // Set up global event listeners
        this.setupGlobalEvents();
        
        // Test backend connection
        this.testBackendConnection();
        
        // Load initial tab
        this.loadTab('srt-processing');
        
        console.log('DracinDub Web App initialized');
    }

    setupTabNavigation() {
        console.log('Setting up tab navigation...');
        
        // Tab click handlers
        document.querySelectorAll('[data-tab]').forEach(tab => {
            tab.addEventListener('click', (e) => {
                e.preventDefault();
                const tabName = tab.getAttribute('data-tab');
                console.log('Tab clicked:', tabName);
                this.loadTab(tabName);
            });
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.ctrlKey || e.metaKey) {
                switch(e.key) {
                    case '1':
                        e.preventDefault();
                        this.loadTab('srt-processing');
                        break;
                    case '2':
                        e.preventDefault();
                        this.loadTab('translate');
                        break;
                    case '3':
                        e.preventDefault();
                        this.loadTab('editing');
                        break;
                    case '4':
                        e.preventDefault();
                        this.loadTab('export');
                        break;
					case '5':
						e.preventDefault();
						this.loadTab('ocr-edit');
						break;	
                }
            }
        });
    }

    setupGlobalEvents() {
        console.log('Setting up global events...');
        
        // Global save handler
        const saveBtn = document.getElementById('save-all');
        if (saveBtn) {
            saveBtn.addEventListener('click', () => {
                this.saveAll();
            });
        }

        const settingsBtn = document.getElementById('settings');
        if (settingsBtn) {
            settingsBtn.addEventListener('click', () => {
                this.showNotification('Settings will be available soon', 'info');
            });
        }

        const helpBtn = document.getElementById('help');
        if (helpBtn) {
            helpBtn.addEventListener('click', () => {
                window.open('https://www.youtube.com/@Dramapendekyangbagus', '_blank');
            });
        }

        // Online/offline detection
        window.addEventListener('online', () => {
            this.showNotification('Connection restored', 'success');
        });

        window.addEventListener('offline', () => {
            this.showNotification('Connection lost', 'error');
        });
    }

	async loadTab(tabName) {
	  console.log('Loading tab:', tabName);

	  // Toggle state di tombol tab
	  document.querySelectorAll('[data-tab]').forEach(tab => {
		tab.classList.remove('active', 'border-blue-400', 'bg-gray-700');
		tab.classList.add('border-transparent');
	  });
	  const activeBtn = document.querySelector(`[data-tab="${tabName}"]`);
	  if (activeBtn) {
		activeBtn.classList.add('active', 'border-blue-400', 'bg-gray-700');
		activeBtn.classList.remove('border-transparent');
	  }

	  // TUTUP semua konten tab (hapus .active)
	  document.querySelectorAll('.tab-content').forEach(el => {
		el.classList.remove('active');
	  });

	  // BUKA konten tab yang dipilih (tambah .active)
	  const tabContent = document.getElementById(tabName);
	  if (tabContent) {
		tabContent.classList.add('active');
	  }
		if (tabName === 'editing') {
		  await this.loadEditingTab();
		  return;
		}

	  this.activeTab = tabName;

	  // Muat konten spesifik tab
	  await this.loadTabContent(tabName);
	}


	async loadTabContent(tabName) {
	  // >>> ADD: jangan render ulang Tab Translate kalau sudah terpasang
	  if (tabName === 'translate') {
		const cont = document.getElementById('translate');
		if (cont && cont.dataset.inited === '1') {
		  // Sudah mounted: cukup aktifkan tab & refresh UI ringan
		  this.activeTab = tabName;
		  this.filterAndRenderSubs?.();
		  this.updateTranslateProgress?.(
			this.translate?.totalDone || 0,
			this.translate?.totalAll  || 0
		  );
		  return; // <-- kunci: SKIP render ulang & SKIP fetch lagi
		}
	  }
	  // <<< END ADD
		this.showLoading(`Loading ${tabName.replace('-', ' ')}...`);
		
		try {
			switch(tabName) {
				case 'srt-processing':
					await this.loadSRTProcessingTab();
					break;
				case 'translate':
					await this.loadTranslateTab();  // Pastikan ini yang dipanggil
					break;
				case 'editing':
					await this.loadEditingTab();
					break;
				case 'export':
					await this.loadExportTab();   // <-- tambahkan baris ini
					break;
				case 'ocr-edit':
					await this.loadOcrEditTab();
					break;
			}
		} catch (error) {
			console.error(`Error loading tab ${tabName}:`, error);
			this.showNotification(`Error loading ${tabName}: ${error.message}`, 'error');
		} finally {
			this.hideLoading();
		}
	}

	async loadSRTProcessingTab() {
	  const tab = document.getElementById('srt-processing');
	  if (!tab) return;
	  tab.innerHTML = `
		<!-- spinner singkat boleh, tapi langsung render saja juga boleh -->
		<div class="bg-gray-800 rounded-lg p-6">
		  <h2 class="text-2xl font-bold mb-6 text-blue-400">
			<i class="fas fa-file-alt mr-2"></i>SRT Processing
		  </h2>
		  <div class="text-center py-8">
			<div class="loading-spinner mx-auto mb-4"></div>
			<p class="text-gray-400">Loading SRT Processing interface...</p>
		  </div>
		</div>
	  `;
	  // langsung render UI + bind
	  this.renderSRTProcessingTab();
	}

    renderSRTProcessingTab() {
        const tabContent = document.getElementById('srt-processing');
        if (!tabContent) return;

	tabContent.innerHTML = `
		<div class="bg-gray-800 rounded-lg p-6">
			<h2 class="text-2xl font-bold mb-6 text-blue-400">
				<i class="fas fa-file-alt mr-2"></i>SRT Processing
			</h2>

			<!-- File Upload Section -->
			<div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
				<!-- Video Upload -->
				<div class="bg-gray-700 rounded-lg p-4">
					<h3 class="text-lg font-semibold mb-3 text-white">
						<i class="fas fa-video mr-2"></i>Video File
					</h3>
					<div class="border-2 border-dashed border-gray-600 rounded-lg p-6 text-center hover:border-blue-400 transition-colors cursor-pointer" id="video-drop-zone">
						<i class="fas fa-cloud-upload-alt text-4xl text-gray-400 mb-3"></i>
						<p class="text-gray-300 mb-2">Drag & drop video file or click to browse</p>
						<p class="text-sm text-gray-400 mb-4">Supported: MP4, MKV, AVI, MOV</p>
						<button class="bg-blue-500 hover:bg-blue-600 px-4 py-2 rounded transition-colors">
							<i class="fas fa-search mr-2"></i>Browse Files
						</button>
						<div id="video-file-info" class="mt-3 text-sm text-green-400 hidden">
							<i class="fas fa-check-circle mr-1"></i>
							<span>Video file selected</span>
						</div>
					</div>
				</div>

				<!-- SRT Upload -->
				<div class="bg-gray-700 rounded-lg p-4">
					<h3 class="text-lg font-semibold mb-3 text-white">
						<i class="fas fa-file-text mr-2"></i>SRT File
					</h3>
					<div class="border-2 border-dashed border-gray-600 rounded-lg p-6 text-center hover:border-blue-400 transition-colors cursor-pointer" id="srt-drop-zone">
						<i class="fas fa-file-upload text-4xl text-gray-400 mb-3"></i>
						<p class="text-gray-300 mb-2">Drag & drop SRT Hasil OCR</p>
						<p class="text-sm text-gray-400 mb-4">Format: SubRip (.srt)</p>
						<button class="bg-blue-500 hover:bg-blue-600 px-4 py-2 rounded transition-colors">
							<i class="fas fa-search mr-2"></i>Browse Files
						</button>
						<div id="srt-file-info" class="mt-3 text-sm text-green-400 hidden">
							<i class="fas fa-check-circle mr-1"></i>
							<span>SRT file selected</span>
						</div>
					</div>
				</div>
			</div>

			<!-- Diarization Configuration -->
			<div class="bg-gray-700 rounded-lg p-4 mb-6">
				<h3 class="text-lg font-semibold mb-4 text-white">
					<i class="fas fa-users mr-2"></i>Diarization Configuration
				</h3>

				<div class="grid grid-cols-1 md:grid-cols-2 gap-4">

					<!-- Gender Mode -->
					<div>
						<label class="block text-sm font-medium mb-2 text-gray-300">Gender Mode</label>
						<select id="gender-mode"
								class="w-full px-3 py-2 bg-gray-600 rounded border border-gray-500 focus:border-blue-500 text-white">
							<option value="hf_svm" selected>Pretrained (HF SVM)</option>
							<option value="reference">Reference Banks (Manual)</option>
						</select>
						<p class="text-xs text-gray-400 mt-1">
							hf_svm = otomatis pakai model pretrained<br>
							reference = pakai folder Male/Female kamu
						</p>
					</div>

					<!-- Top N Segments -->
					<div>
						<label class="block text-sm font-medium mb-2 text-gray-300">Top N Segments</label>
						<input id="top-n" type="number" value="6"
							   class="w-full px-3 py-2 bg-gray-600 rounded border border-gray-500 focus:border-blue-500 text-white">
						<p class="text-xs text-gray-400 mt-1">Ambil segmen terpanjang speaker untuk analisis gender</p>
					</div>

					<!-- Male/Female ref (MODE: reference) -->
					<div id="ref-male-wrap">
						<label class="block text-sm font-medium mb-2 text-gray-300">Male Reference</label>
						<input id="male-ref" type="text" value="D:\\dubdracin\\samples\\male"
							   class="w-full px-3 py-2 bg-gray-600 rounded border border-gray-500 focus:border-blue-500 text-white placeholder-gray-400">
					</div>

					<div id="ref-female-wrap">
						<label class="block text-sm font-medium mb-2 text-gray-300">Female Reference</label>
						<input id="female-ref" type="text" value="D:\\dubdracin\\samples\\female"
							   class="w-full px-3 py-2 bg-gray-600 rounded border border-gray-500 focus:border-blue-500 text-white placeholder-gray-400">
					</div>

					<!-- HF SVM settings (MODE: hf_svm) -->
					<div id="hf-minvote-wrap">
						<label class="block text-sm font-medium mb-2 text-gray-300">min_vote</label>
						<input id="min-vote" type="number" min="0.5" max="1.0" step="0.05" value="0.7"
							   class="w-full px-3 py-2 bg-gray-600 rounded border border-gray-500 focus:border-blue-500 text-white">
						<p class="text-xs text-gray-400 mt-1">
							Mayoritas suara yg sama harus ≥ nilai ini (kalau tidak → Unknown)
						</p>
					</div>

					<div id="hf-minlen-wrap">
						<label class="block text-sm font-medium mb-2 text-gray-300">min_len_sec</label>
						<input id="min-len-sec" type="number" min="0.2" step="0.1" value="0.5"
							   class="w-full px-3 py-2 bg-gray-600 rounded border border-gray-500 focus:border-blue-500 text-white">
						<p class="text-xs text-gray-400 mt-1">
							Abaikan potongan suara yang terlalu pendek (teriak sepersekian detik)
						</p>
					</div>

					<!-- HF Token -->
					<div class="md:col-span-2">
						<label class="block text-sm font-medium mb-2 text-gray-300">HF Token</label>
						<input id="hf-token" type="password" value="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"
							   class="w-full px-3 py-2 bg-gray-600 rounded border border-gray-500 focus:border-blue-500 text-white placeholder-gray-400">
					</div>

					<!-- —— Global Linking Settings —— -->
					<div class="col-span-1 md:col-span-2 border-t border-gray-600 pt-3 mt-2"></div>

					<div class="flex items-center">
						<input id="link-global" type="checkbox" class="mr-2" checked>
						<label for="link-global" class="text-gray-300">Enable Global Speaker Linking</label>
					</div>

					<div>
						<label class="block text-sm font-medium mb-2 text-gray-300">link_threshold</label>
						<input id="link-threshold" type="number" min="0" max="1" step="0.01" value="0.93"
							   class="w-full px-3 py-2 bg-gray-600 rounded border border-gray-500 text-white">
					</div>

					<div>
						<label class="block text-sm font-medium mb-2 text-gray-300">samples_per_spk</label>
						<input id="samples-per-spk" type="number" min="1" step="1" value="8"
							   class="w-full px-3 py-2 bg-gray-600 rounded border border-gray-500 text-white">
					</div>

					<div>
						<label class="block text-sm font-medium mb-2 text-gray-300">min_speakers (opsional)</label>
						<input id="min-speakers" type="number" min="1" step="1"
							   class="w-full px-3 py-2 bg-gray-600 rounded border border-gray-500 text-white">
					</div>

					<div>
						<label class="block text-sm font-medium mb-2 text-gray-300">max_speakers (opsional)</label>
						<input id="max-speakers" type="number" min="1" step="1"
							   class="w-full px-3 py-2 bg-gray-600 rounded border border-gray-500 text-white">
					</div>

					<div>
						<label class="block text-sm font-medium mb-2 text-gray-300">min_sample_dur (detik)</label>
						<input id="min-sample-dur" type="number" min="0.2" step="0.1" value="1.5"
							   class="w-full px-3 py-2 bg-gray-600 rounded border border-gray-500 text-white">
					</div>

					<div class="flex items-center mt-2 md:col-span-2">
						<input id="use-gpu" type="checkbox" checked class="mr-2 w-4 h-4 text-blue-500 bg-gray-600 border-gray-500 rounded focus:ring-blue-400">
						<label for="use-gpu" class="text-gray-300">Use GPU for Diarization</label>
					</div>
				</div>
			</div>

			<!-- Progress Section -->
			<div class="bg-gray-700 rounded-lg p-4 mb-6">
				<h3 class="text-lg font-semibold mb-3 text-white">
					<i class="fas fa-tasks mr-2"></i>Processing Progress
				</h3>
				<div class="space-y-4">
					<div>
						<div class="flex justify-between mb-2">
							<span class="text-sm text-gray-300" id="current-step">Ready to process</span>
							<span class="text-sm text-gray-300" id="progress-percent">0%</span>
						</div>
						<div class="w-full bg-gray-600 rounded-full h-3">
							<div id="progress-bar" class="bg-blue-500 h-3 rounded-full transition-all duration-500 ease-out" style="width: 0%"></div>
						</div>
					</div>
				</div>
			</div>

			<!-- Action Buttons -->
			<div class="flex flex-wrap gap-3">
				<button id="create-session" class="bg-green-500 hover:bg-green-600 px-4 py-2 rounded flex items-center transition-colors">
					<i class="fas fa-plus mr-2"></i>Create Session
				</button>
				<button id="generate-workdir" class="bg-blue-500 hover:bg-blue-600 px-4 py-2 rounded flex items-center transition-colors">
					<i class="fas fa-folder-plus mr-2"></i>Generate Workdir
				</button>
				<button id="extract-audio" class="bg-purple-500 hover:bg-purple-600 px-4 py-2 rounded flex items-center transition-colors">
					<i class="fas fa-volume-up mr-2"></i>Extract Audio
				</button>
				<button id="run-diarization" class="bg-orange-500 hover:bg-orange-600 px-4 py-2 rounded flex items-center transition-colors">
					<i class="fas fa-users mr-2"></i>Run Diarization
				</button>
				<button id="load-to-translate" class="bg-indigo-500 hover:bg-indigo-600 px-4 py-2 rounded flex items-center transition-colors">
					<i class="fas fa-arrow-right mr-2"></i>Load to Translate
				</button>
			</div>

			<!-- Load / Delete Existing Session -->
			<div class="mt-4 bg-gray-700 rounded-lg p-4">
				<h3 class="text-lg font-semibold mb-3 text-white">
					<i class="fas fa-folder-open mr-2"></i>Manage Sessions
				</h3>

				<div class="grid grid-cols-1 md:grid-cols-3 gap-4 items-end">
					<!-- dropdown pilih session -->
					<div class="md:col-span-2">
						<label class="block text-sm font-medium mb-2 text-gray-300">Existing Sessions</label>
						<select id="proc-session-select"
							class="w-full px-3 py-2 bg-gray-600 rounded border border-gray-500 focus:border-blue-500 text-white">
							<option value="">— pilih session —</option>
						</select>
					</div>

					<!-- tombol load -->
					<div class="flex gap-2">
						<button id="proc-load-session"
							class="bg-yellow-500 hover:bg-yellow-600 px-4 py-2 rounded text-sm transition-colors">
							<i class="fas fa-download mr-1"></i>Load
						</button>

						<button id="proc-delete-session"
							class="bg-red-600 hover:bg-red-700 px-4 py-2 rounded text-sm transition-colors">
							<i class="fas fa-trash mr-1"></i>Delete
						</button>
					</div>
				</div>
			</div>


			<!-- Log Output -->
			<div class="mt-6">
				<h3 class="text-lg font-semibold mb-3 text-white">
					<i class="fas fa-terminal mr-2"></i>Processing Log
				</h3>
				<div id="log-output" class="bg-gray-900 rounded p-4 h-48 overflow-y-auto font-mono text-sm">
					<div class="text-gray-500">Log output will appear here...</div>
				</div>
			</div>
		</div>
	`;

	// setelah render HTML tab SRT Processing:
	this._procPopulateSessionSelector();

	// tombol load session lama
	document
	  .getElementById('proc-load-session')
	  ?.addEventListener('click', () => this._procLoadSelectedSession());

	// tombol delete session
	document
	  .getElementById('proc-delete-session')
	  ?.addEventListener('click', () => this._procDeleteSelectedSession());


	// tampil/sembunyikan blok input sesuai gender mode
	function toggleGenderModeUI() {
		const mode = document.getElementById('gender-mode')?.value || 'hf_svm';
		const showRef = (mode === 'reference');

		// mode reference butuh male/female ref, sembunyikan setting hf_svm
		document.getElementById('ref-male-wrap').style.display   = showRef ? '' : 'none';
		document.getElementById('ref-female-wrap').style.display = showRef ? '' : 'none';

		// mode hf_svm butuh min_vote/min_len_sec
		document.getElementById('hf-minvote-wrap').style.display = showRef ? 'none' : '';
		document.getElementById('hf-minlen-wrap').style.display  = showRef ? 'none' : '';
	}

	// pas pertama render SRT tab:
	toggleGenderModeUI();
	document.getElementById('gender-mode')?.addEventListener('change', toggleGenderModeUI);


        // Add event listeners for this tab
        this.setupSRTProcessingEvents();
    }

    setupSRTProcessingEvents() {
        console.log('Setting up SRT Processing events...');
        
        // Create test session button
        const createTestBtn = document.getElementById('create-test-session');
        if (createTestBtn) {
            createTestBtn.addEventListener('click', () => {
                this.createTestSession();
            });
        }

        // Create session button
        const createSessionBtn = document.getElementById('create-session');
        if (createSessionBtn) {
          createSessionBtn.addEventListener('click', () => {
            this.createSessionWithFiles();
          });
        }

        // Other action buttons
        const actionButtons = ['generate-workdir', 'extract-audio', 'run-diarization', 'load-to-translate'];
        actionButtons.forEach(btnId => {
            const btn = document.getElementById(btnId);
            if (btn) {
                btn.addEventListener('click', () => {
                    this.handleSRTAction(btnId);
                });
            }
        });

        // File drop zones - FIXED VERSION
        this.setupFileDropZones();
    }

    setupFileDropZones() {
        // Create hidden file inputs
        this.createFileInput('video', ['video/*', '.mp4', '.mkv', '.avi', '.mov']);
        this.createFileInput('srt', ['.srt', '.txt']);
        
        // Video drop zone
        const videoDropZone = document.getElementById('video-drop-zone');
        if (videoDropZone) {
            // Click handler
            videoDropZone.addEventListener('click', () => {
                document.getElementById('video-file-input').click();
            });
            
            // Drag and drop handlers
            this.setupDragAndDrop(videoDropZone, 'video');
        }

        // SRT drop zone  
        const srtDropZone = document.getElementById('srt-drop-zone');
        if (srtDropZone) {
            // Click handler
            srtDropZone.addEventListener('click', () => {
                document.getElementById('srt-file-input').click();
            });
            
            // Drag and drop handlers
            this.setupDragAndDrop(srtDropZone, 'srt');
        }
    }

    createFileInput(type, accept) {
        // Remove existing input if any
        const existingInput = document.getElementById(`${type}-file-input`);
        if (existingInput) {
            existingInput.remove();
        }
        
        // Create new file input
        const fileInput = document.createElement('input');
        fileInput.type = 'file';
        fileInput.id = `${type}-file-input`;
        fileInput.accept = accept.join(',');
        fileInput.style.display = 'none';
        
        fileInput.addEventListener('change', (e) => {
            this.handleFileSelect(e, type);
        });
        
        document.body.appendChild(fileInput);
    }

	// isi dropdown di tab SRT Processing
	async _procPopulateSessionSelector() {
	  try {
		const res = await fetch('/api/sessions');
		if (!res.ok) {
		  this.showNotification('Gagal ambil daftar session', 'error');
		  return;
		}

		const list = await res.json(); // backend balikin array session [{id,...}] :contentReference[oaicite:3]{index=3}
		const sel = document.getElementById('proc-session-select');
		if (!sel) return;

		sel.innerHTML =
		  `<option value="">— pilih session —</option>` +
		  list.map(s => `<option value="${s.id}">${s.id}</option>`).join('');
	  } catch (err) {
		console.error(err);
	  }
	}

	// klik tombol "Load" di tab SRT Processing
	async _procLoadSelectedSession() {
	  const sel = document.getElementById('proc-session-select');
	  const chosen = sel ? sel.value.trim() : '';
	  if (!chosen) {
		this.showNotification('Pilih session dulu', 'warning');
		return;
	  }

	  // set session aktif global app
	  this.currentSessionId = chosen;

	  // update tampilan "Current Session" card
	  this.updateSessionInfo(chosen);

	  this.showNotification(`Session ${chosen} loaded`, 'success');
	}

	// klik tombol "Delete" (opsional, aktif kalau backend delete sudah ada)
	async _procDeleteSelectedSession() {
	  const sel = document.getElementById('proc-session-select');
	  const chosen = sel ? sel.value.trim() : '';
	  if (!chosen) {
		this.showNotification('Pilih session yang mau dihapus', 'warning');
		return;
	  }

	  // konfirmasi kecil di UI
	  if (!confirm(`Hapus session ${chosen}? (folder kerja akan hilang)`)) {
		return;
	  }

	  // panggil backend delete (lihat poin 3 di bawah)
	  const res = await fetch(`/api/session/${chosen}/delete`, {
		method: 'POST'
	  });

	  const txt = await res.text();
	  if (!res.ok) {
		this.showNotification(`Gagal hapus: ${txt}`, 'error');
		return;
	  }

	  this.showNotification(`Session ${chosen} dihapus`, 'success');

	  // kalau session yang lagi aktif kebetulan sama -> kosongkan
	  if (this.currentSessionId === chosen) {
		this.currentSessionId = null;
		this.updateSessionInfo('(no active session)');
	  }

	  // refresh isi dropdown supaya langsung hilang tanpa Ctrl+F5
	  this._procPopulateSessionSelector();
	}


    setupDragAndDrop(dropZone, type) {
        // Drag over effect
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('border-blue-400', 'bg-gray-600');
        });

        // Drag leave effect
        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('border-blue-400', 'bg-gray-600');
        });

        // Drop handler
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('border-blue-400', 'bg-gray-600');
            
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleDroppedFile(files[0], type);
            }
        });
    }

    handleFileSelect(event, type) {
        const file = event.target.files[0];
        if (file) {
            this.handleDroppedFile(file, type);
        }
    }

    handleDroppedFile(file, type) {
        const fileInfo = document.getElementById(`${type}-file-info`);
        const fileName = file.name;
        const fileSize = (file.size / (1024 * 1024)).toFixed(2);

        // Validate file type
        if (type === 'video' && !file.type.startsWith('video/') && !fileName.match(/\.(mp4|mkv|avi|mov)$/i)) {
            this.showNotification('Please select a valid video file (MP4, MKV, AVI, MOV)', 'error');
            return;
        }
        
        if (type === 'srt' && !fileName.match(/\.(srt|txt)$/i)) {
            this.showNotification('Please select a valid SRT file (.srt)', 'error');
            return;
        }

        // Update UI
        fileInfo.innerHTML = `
            <div class="text-green-400">
                <i class="fas fa-check-circle mr-1"></i>
                ${fileName} (${fileSize} MB)
            </div>
        `;
        fileInfo.classList.remove('hidden');

        // Store file reference
        if (type === 'video') {
            this.videoFile = file;
        } else {
            this.srtFile = file;
        }

        this.showNotification(`${type.toUpperCase()} file selected: ${fileName}`, 'success');
        this.updateButtonStates();
    }

    updateButtonStates() {
        const hasFiles = this.videoFile && this.srtFile;
        const createSessionBtn = document.getElementById('create-session');
        
        if (createSessionBtn) {
            createSessionBtn.disabled = !hasFiles;
            if (hasFiles) {
                createSessionBtn.classList.remove('opacity-50', 'cursor-not-allowed');
            } else {
                createSessionBtn.classList.add('opacity-50', 'cursor-not-allowed');
            }
        }
    }

    handleSRTAction(action) {
        console.log(`🔄 SRT Action: ${action}`);
        
        if (!this.currentSessionId) {
            this.showNotification('Please create a session first', 'error');
            return;
        }

        const actions = {
            'generate-workdir': () => this.generateWorkdir(),
            'extract-audio': () => this.extractAudio(),
            'run-diarization': () => this.runDiarization(),
            'load-to-translate': () => this.loadToTranslate()
        };

        if (actions[action]) {
            actions[action]();
        }
    }

    async generateWorkdir() {
        try {
            this.showLoading('Generating workspace...');
            const response = await fetch(`/api/session/${this.currentSessionId}/generate-workdir`, {
                method: 'POST'
            });
            
            if (response.ok) {
                const data = await response.json();
                this.showNotification('Workspace generated successfully', 'success');
                this.updateProgress(20, 'Workspace ready');
            } else {
                throw new Error('Failed to generate workdir');
            }
        } catch (error) {
            this.showNotification(`Error: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }

	async extractAudio() {
	  try {
		this.showLoading('Extracting audio...');
		const body = new URLSearchParams({
		  vocal_only: 'true',      // paksa vokal saja
		  prefer: 'demucs'         // jangan biarkan auto jatuh ke ffmpeg_mid
		});

		const res = await fetch(`/api/session/${this.currentSessionId}/extract-audio`, {
		  method: 'POST',
		  headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
		  body
		});

		const text = await res.text();
		if (!res.ok) {
		  this.showNotification(`Extract failed: ${text}`, 'error');
		  return;
		}

		let data = {};
		try { data = JSON.parse(text); } catch {}
		const iso = data.vocal_isolation || 'unknown';
		this.showNotification(`Audio extracted (${iso})`, 'success');
		this.updateProgress(40, 'Audio extracted');
		this.appendLog?.(JSON.stringify(data));
	  } catch (error) {
		this.showNotification(`Error: ${error.message}`, 'error');
	  } finally {
		this.hideLoading();
	  }
	}


	// Tab 1 – Diarization
	async runDiarization() {
	  if (!this.currentSessionId) {
		this.showNotification('Create Session dulu.', 'error');
		return;
	  }

	  // --- ambil semua elemen UI ---
	  const modeEl        = document.getElementById('gender-mode');
	  const maleRefEl     = document.getElementById('male-ref');
	  const femaleRefEl   = document.getElementById('female-ref');
	  const minVoteEl     = document.getElementById('min-vote');
	  const minLenEl      = document.getElementById('min-len-sec');

	  const topNEl        = document.getElementById('top-n');
	  const hfTokenEl     = document.getElementById('hf-token');
	  const useGpuEl      = document.getElementById('use-gpu');

	  const linkGlobalEl    = document.getElementById('link-global');
	  const linkThresholdEl = document.getElementById('link-threshold');
	  const spsEl           = document.getElementById('samples-per-spk');
	  const minSpeakersEl   = document.getElementById('min-speakers');
	  const maxSpeakersEl   = document.getElementById('max-speakers');
	  const minSampleDurEl  = document.getElementById('min-sample-dur');

	  // --- ambil nilai ---
	  const gender_mode   = (modeEl?.value || 'hf_svm').trim(); // "hf_svm" / "reference"
	  const male_ref      = (maleRefEl?.value || '').trim();
	  const female_ref    = (femaleRefEl?.value || '').trim();

	  const min_vote      = (minVoteEl?.value || '0.7').trim();
	  const min_len_sec   = (minLenEl?.value || '0.5').trim();

	  const top_n         = topNEl?.value ? parseInt(topNEl.value, 10) : 6;
	  const hf_token      = (hfTokenEl?.value || '').trim();
	  const use_gpu       = useGpuEl?.checked ? 'true' : 'false';

	  const link_global     = linkGlobalEl?.checked ? 'true' : 'false';
	  const link_threshold  = linkThresholdEl?.value || '0.93';
	  const samples_per_spk = spsEl?.value || '8';
	  const min_speakers    = minSpeakersEl?.value || '';
	  const max_speakers    = maxSpeakersEl?.value || '';
	  const min_sample_dur  = minSampleDurEl?.value || '1.5';

	  // --- validasi ringan ---
	  if (gender_mode === 'reference') {
		if (!male_ref || !female_ref) {
		  this.showNotification('Male/Female Reference wajib diisi untuk mode Reference.', 'error');
		  return;
		}
	  }

	  if (!hf_token) {
		this.showNotification('HF Token wajib diisi.', 'error');
		return;
	  }

	  // --- siapkan body request sesuai backend baru kita ---
	  const body = new URLSearchParams();
	  body.append('gender_mode', gender_mode);
	  body.append('hf_token', hf_token);
	  body.append('use_gpu', use_gpu);
	  body.append('top_n', String(top_n));

	  if (gender_mode === 'reference') {
		body.append('male_ref', male_ref);
		body.append('female_ref', female_ref);
		// (mode lama tidak butuh min_vote/min_len_sec)
	  } else {
		// hf_svm mode
		body.append('min_vote', String(min_vote));
		body.append('min_len_sec', String(min_len_sec));
		// (hf_svm tidak butuh male_ref/female_ref)
	  }

	  // global linking
	  body.append('link_global', link_global);
	  body.append('link_threshold', String(link_threshold));
	  body.append('samples_per_spk', String(samples_per_spk));
	  if (min_speakers) body.append('min_speakers', String(parseInt(min_speakers, 10)));
	  if (max_speakers) body.append('max_speakers', String(parseInt(max_speakers, 10)));
	  body.append('min_sample_dur', String(min_sample_dur));

	  const url = `/api/session/${this.currentSessionId}/diarization`;

	  try {
		this.appendLog?.('Running diarization...');
		this.showLoading('Running diarization...');

		const res = await fetch(url, {
		  method: 'POST',
		  headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
		  body,
		});

		const text = await res.text();

		if (!res.ok) {
		  this.showNotification(`Failed to run diarization: ${text}`, 'error');
		  this.appendLog?.(text);
		  return;
		}

		let data = {};
		try { data = JSON.parse(text); } catch (_) {}

		const segPath = data.segments_path || data.segjson || '';
		const spkPath = data.speakers_path || data.spkjson || '';

		if (segPath) this.appendLog?.(`Segments: ${segPath}`);
		if (spkPath) this.appendLog?.(`Speakers: ${spkPath}`);

		this.updateProgress?.(100, 'Diarization done');
		this.showNotification('Diarization completed successfully', 'success');
	  } catch (err) {
		this.showNotification(`Failed to run diarization: ${err?.message || err}`, 'error');
	  } finally {
		this.hideLoading();
	  }
	}


    updateProgress(percent, message) {
      const bar   = document.getElementById('progress-bar');
      const pctEl = document.getElementById('progress-percent');
      const step  = document.getElementById('current-step');

      if (bar)   bar.style.width = `${Number(percent)||0}%`;
      if (pctEl) pctEl.textContent = `${Number(percent)||0}%`;
      if (step)  step.textContent = message || '';
    }

    loadToTranslate() {
        this.showNotification('Loading to Translate tab...', 'info');
        setTimeout(() => {
            this.loadTab('translate');
        }, 1000);
    }

    async createSessionWithFiles() {
        if (!this.videoFile || !this.srtFile) {
            this.showNotification('Please select both video and SRT files first', 'error');
            return;
        }

        this.showLoading('Creating session and uploading files...');

        try {
            // First create session
            const sessionResponse = await fetch('/api/session/create', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: `video_name=${encodeURIComponent(this.videoFile.name)}&srt_name=${encodeURIComponent(this.srtFile.name)}`
            });

            if (!sessionResponse.ok) {
                throw new Error('Failed to create session');
            }

            const sessionData = await sessionResponse.json();
            this.currentSessionId = sessionData.session_id;
            
            // Then upload files
            const formData = new FormData();
            formData.append('video', this.videoFile);
            formData.append('srt', this.srtFile);

            const uploadResponse = await fetch(`/api/session/${this.currentSessionId}/upload`, {
                method: 'POST',
                body: formData
            });

            if (!uploadResponse.ok) {
                throw new Error('Failed to upload files');
            }

            this.showNotification('Session created and files uploaded successfully!', 'success');
            this.updateSessionInfo(this.currentSessionId);
            
            // Simulate progress
            this.simulateProgress();

        } catch (error) {
            console.error('Error creating session:', error);
            this.showNotification(`Error: ${error.message}`, 'error');
        } finally {
            this.hideLoading();
        }
    }

    simulateProgress() {
        let progress = 0;
        const progressBar = document.getElementById('progress-bar');
        const progressPercent = document.getElementById('progress-percent');
        const currentStep = document.getElementById('current-step');
        
        const steps = [
            'Creating session...',
            'Uploading files...', 
            'Processing files...',
            'Ready for next steps'
        ];
        
        const interval = setInterval(() => {
            progress += 25;
            
            if (progressBar) progressBar.style.width = `${progress}%`;
            if (progressPercent) progressPercent.textContent = `${progress}%`;
            if (currentStep) {
                currentStep.textContent = steps[Math.min(Math.floor(progress / 25), steps.length - 1)];
            }
            
            if (progress >= 100) {
                clearInterval(interval);
                this.showNotification('Files processed successfully!', 'success');
                
                // Enable other buttons
                const buttons = ['generate-workdir', 'extract-audio', 'run-diarization', 'load-to-translate'];
                buttons.forEach(btnId => {
                    const btn = document.getElementById(btnId);
                    if (btn) {
                        btn.disabled = false;
                        btn.classList.remove('opacity-50', 'cursor-not-allowed');
                    }
                });
            }
        }, 800);
    }

// Perbaiki method loadTranslateTab() untuk auto-load SRT
	async loadTranslateTab() {
	  const tab = document.getElementById('translate');
	  if (!tab) return;
	  tab.innerHTML = `
		<div class="bg-gray-800 rounded-lg p-6">
		  <h2 class="text-2xl font-bold mb-6 text-green-400">
			<i class="fas fa-language mr-2"></i>Translate
		  </h2>
		  <div class="text-center py-8">
			<div class="loading-spinner mx-auto mb-4"></div>
			<p class="text-gray-400">Loading Translation interface...</p>
		  </div>
		</div>
	  `;

	  // renderTranslateTab() SUDAH memanggil setupTranslateEvents() di dalamnya
	  this.renderTranslateTab();

	// Auto-load hanya jika BELUM ada data dan TIDAK sedang streaming
	const st = this.translate || (this.initTranslateState(), this.translate);
	if ((!Array.isArray(st.subtitles) || st.subtitles.length === 0) && !st.running) {
	  if (this.currentSessionId) {
		try {
		  const res = await fetch(`/api/session/${this.currentSessionId}/srt?prefer=original`);
		  if (res.ok) {
			let text = await res.text();
			// kalau backend kirim {srt: "..."} tetap dukung
			try { const obj = JSON.parse(text); if (obj && obj.srt) text = obj.srt; } catch {}
			const items = this.parseSRT(text);
			st.originalSubs = items.map(o => ({ ...o }));
			st.subtitles    = items.map(o => ({ ...o, trans: '' }));
			this.filterAndRenderSubs?.();
			this.showNotification('SRT loaded from session', 'success');
		  }
		} catch {}
	  }
	}
}

// === REPLACE: appendMessage + renderMessages ===
appendMessage(obj) {
  if (!this.translate) this.initTranslateState();
  const arr = this.translate.messages || (this.translate.messages = []);
  arr.push({ ts: new Date().toISOString(), ...obj });
  if (arr.length > 500) arr.splice(0, arr.length - 500);   // jaga memori
  this.renderMessages(true);
}

// === UI TYPEWRITER: lambat & natural ===
_typeToInput(inputEl, finalText, cps = 18, onDone) {
  if (!inputEl) { if (onDone) onDone(); return; }

  // basis: ms per karakter dari cps
  const base = Math.floor(1000 / Math.max(1, cps));
  const len  = (finalText || "").length;

  // adaptif: kalimat pendek → lebih lambat biar terasa "ngetik"
  let msPerChar = base;
  if (len < 20)      msPerChar = Math.floor(base * 1.8);
  else if (len < 40) msPerChar = Math.floor(base * 1.4);

  // batas aman
  msPerChar = Math.min(200, Math.max(15, msPerChar));

  // jitter kecil biar tidak seragam banget
  const jitterPct = 0.15;

  // hentikan typing sebelumnya pada input yang sama
  if (inputEl._twTimer) clearTimeout(inputEl._twTimer);
  inputEl.value = "";
  let i = 0;

  const step = () => {
	if (!inputEl.isConnected) {
	  // elemen hilang (pindah tab / re-render). WAJIB resolve agar _drainStreamEvents tidak hang.
	  if (inputEl._twTimer) clearTimeout(inputEl._twTimer);
	  inputEl._twTimer = null;
	  if (onDone) onDone();
	  return;
	}
    inputEl.value += (finalText[i++] || "");
    if (i < len) {
      const jitter = Math.floor(msPerChar * jitterPct * Math.random());
      inputEl._twTimer = setTimeout(step, msPerChar + jitter);
    } else {
      inputEl._twTimer = null;
      if (onDone) onDone();
    }
  };
  step();
}

_sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

_enqueueStreamEvent(evt) {
  if (!this.translate) this.initTranslateState();
  this.translate.evQueue.push(evt);
  if (!this.translate.evtProcessing) this._drainStreamEvents();
}

// proses event satu-per-satu; jangan panggil renderSubsPage tiap event
async _drainStreamEvents() {
  const st = this.translate;
  st.evtProcessing = true;
  try {
    while (st.evQueue.length) {
      const evt = st.evQueue.shift();

      // Debug JSON ikut menetes
      this.appendMessage?.(evt);

      if (evt.type === 'progress') {
        const total = (st.subtitles || []).length || evt.total || 0;
        const done  = evt.done || 0;
        const bar = document.getElementById('ts-progressbar');
        const lab = document.getElementById('ts-progresslabel');
        const pct = total ? Math.round(done*100/total) : 0;
        if (bar) bar.style.width = `${pct}%`;
        if (lab) lab.textContent = `${pct}% (${done}/${total})`;
        continue;
      }

      if (evt.type === 'result') {
        const idx  = Number(evt.index);
        const text = (evt.translation || evt.text || '').trim();

        // update state
        const row = st.subtitles?.find(r => r.index === idx);
        if (row) row.trans = text;

        // update DOM baris ini saja (tanpa re-render tabel)
        const sel = `#ts-row-${idx} .ts-trans`;
        let input = document.querySelector(sel);

        // jika input belum ada, coba scroll lalu ambil lagi
        if (!input) {
          this._scrollToTsRow?.(idx);
          await this._sleep(0);
          input = document.querySelector(sel);
        }

        if (input && st.typingEnabled && text.length > 0 && text.length <= (st.typingMaxLen || 200)) {
          await new Promise(res => this._typeToInput(input, text, st.typingCps || 12, res));
        } else if (input) {
          input.value = text;
        }

        this.updateTranslateProgress?.();
        this._flashTsRow?.(idx);
        this._scrollToTsRow?.(idx);

        // jeda kosmetik antar-item dari GUI
        const d = Number(st.typingDelayMs || 0);
        if (d > 0) await this._sleep(d);
        continue;
      }

      // event lain (begin/end/error) — hanya ditampilkan
    }
  } finally {
    st.evtProcessing = false;
  }
}

/** Highlight baris sebentar supaya kelihatan row yang update */
_flashTsRow(idx) {
  const row = document.getElementById(`ts-row-${idx}`);
  if (!row) return;
  row.classList.add('ring-2','ring-emerald-400');
  setTimeout(()=> row.classList.remove('ring-2','ring-emerald-400'), 900);
}

_scrollToTsRow(idx) {
  const row = document.getElementById(`ts-row-${idx}`);
  if (row && this.translate?.followSubs !== false) {
    row.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }
}

renderMessages(newItem = false) {
  const el = document.getElementById('ts-messages');
  if (!el) return;

  // auto-scroll kalau user aktifkan followResults
  const shouldStickBottom =
    this.translate.followResults && (newItem || this._isNearBottom(el));

  // tampilkan 50 event terakhir biar terasa "streaming"
  const last = (this.translate.messages || []).slice(-50);
  el.textContent = JSON.stringify(last, null, 2);

  if (shouldStickBottom) el.scrollTop = el.scrollHeight;
}


    //// SRT parse/format ////
    parseSRT(text) {
      const lines = text.replace(/\r/g,'').split('\n');
      const items = [];
      let i=0;
      while (i < lines.length) {
        const idx = Number(lines[i].trim());
        if (!Number.isInteger(idx)) { i++; continue; }
        const times = lines[i+1]||'';
        const m = times.match(/(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})/);
        if (!m) { i++; continue; }
        i+=2;
        const buf=[];
        while (i<lines.length && lines[i].trim()!=='') { buf.push(lines[i]); i++; }
        items.push({index: idx, start: m[1], end: m[2], text: buf.join('\n')});
        while (i<lines.length && lines[i].trim()==='') i++;
      }
      return items;
    }

    escapeHtml(s){ 
      return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); 
    }

	// PATCH untuk app.js - Ganti fungsi-fungsi Translate

	/* ==============================
	 * TRANSLATE STATE
	 * ============================== */
	initTranslateState() {
	  this.translate = {
		mode: "auto",
		// data
		originalSubs: [],      // simpan original (tetap)
		subtitles: [],         // working copy + tempat menaruh .trans
		filtered: [],
		srtText: "",
		// ui
		page: 1,
		size: 7000,
		follow: true,
		search: "",
		followSubs: true,
		followResults: true,
		// run cfg
		apiKey: "sk-dbb007f10bc34473a2a890d556581edc",
		lang: "id",
		engine: "llm",
		temperature: 0.1,
		top_p: 0.3,
		batch: 30,
		workers: 1,
		timeout: 120,
		typingEnabled: true,
		typingCps: 35,        // ↓ pelan (~83 ms/karakter). Naikkan untuk lebih cepat.
		typingMaxLen: 200,    // kalau >200 char, tempel instan (biar tidak terlalu lama)
		typingDelayMs: 5,    // ↑ jeda antar item dari server (ms). 50–120 enak.
		autosave: true,
		// di dalam objek this.translate
		evQueue: [],          // antrian event dari stream
		evtProcessing: false, // sedang memproses antrian atau tidak
		// exec
		running: false,
		abort: null,
		style: "dubbing",  // "dubbing" | "normal"
		// log
		messages: []
	  };
	}
	
	_isNearBottom(el, pad = 24) {
	  return (el.scrollHeight - el.clientHeight - el.scrollTop) <= pad;
	}
	/* ==============================
	 * RENDER TAB TRANSLATE (layout 1 baris untuk subtitle)
	 * ============================== */
	renderTranslateTab() {
	  if (!this.translate) this.initTranslateState();
	  const st = this.translate;
	  
	  const tabContent = document.getElementById('translate');
	  if (!tabContent) return;

	  tabContent.innerHTML = `
		<div class="bg-gray-800 rounded-lg p-6">
		  <h2 class="text-2xl font-bold mb-6 text-green-400">
			<i class="fas fa-language mr-2"></i>Translate
		  </h2>
		  
		  <div class="p-3 space-y-3">
			<!-- Header dengan controls -->
			<div class="flex flex-wrap items-center gap-3">
			  <div class="flex items-center gap-2">
				<span class="text-sm whitespace-nowrap text-gray-300">API Key</span>
				<input id="ts-api" type="password" value="${st.apiKey || ''}"
					   class="border border-gray-600 bg-gray-700 text-white rounded px-3 py-1 w-[200px]" placeholder="sk-..." />
			  </div>
			  <div class="flex items-center gap-2">
				<span class="text-sm text-gray-300">Batch</span>
				<input id="ts-batch" type="number" min="1" value="${st.batch}"
					   class="border border-gray-600 bg-gray-700 text-white rounded px-2 py-1 w-12" />
			  </div>
			  <div class="flex items-center gap-2">
				<span class="text-sm text-gray-300">Workers</span>
				<input id="ts-workers" type="number" min="1" value="${st.workers}"
					   class="border border-gray-600 bg-gray-700 text-white rounded px-2 py-1 w-10" />
			  </div>
			  <label class="inline-flex items-center gap-1">
				<input id="ts-autosave" type="checkbox" ${st.autosave ? 'checked' : ''} class="w-4 h-4 text-blue-500 bg-gray-600 border-gray-500 rounded focus:ring-blue-400" />
				<span class="text-sm text-gray-300">Autosave</span>
			  </label>

			  <!-- Progress bar -->
			  <div class="ml-2 w-[120px] h-2 bg-gray-700 rounded overflow-hidden">
				<div id="ts-progressbar" class="h-full bg-blue-500 transition-all duration-300" style="width:0%"></div>
			  </div>
			  <div id="ts-progresslabel" class="text-xs text-gray-300 min-w-[50px]">0% (0/0)</div>
			</div>

			<!-- Action buttons & config -->
			<div class="flex flex-wrap items-center gap-2">
			  <div class="flex gap-1 flex-wrap">
				<button id="ts-load" class="btn btn-slate text-xs px-2 py-1">📂 Load</button>
				<button id="ts-start" class="btn btn-primary text-xs px-2 py-1">▶ Start</button>
				<button id="ts-resume" class="btn btn-slate text-xs px-2 py-1">⏵ Resume</button>
				<button id="ts-fill" class="btn btn-slate text-xs px-2 py-1">✳ Fill</button>
				<button id="ts-save" class="btn btn-slate text-xs px-2 py-1">💾 Save</button>
				<button id="ts-export" class="btn btn-slate text-xs px-2 py-1">⬇ Export</button>
				<button id="ts-stop" class="btn btn-danger text-xs px-2 py-1">■ Stop</button>
				<button id="ts-next" class="btn btn-slate text-xs px-2 py-1">➡ Next</button>
				<button id="ts-clear" class="btn btn-slate text-xs px-2 py-1">🗑 Clear</button>
			  </div>
			  <div class="ml-auto flex items-center gap-2 flex-wrap">
				<span class="text-sm text-gray-300">Style</span>
				<select id="ts-style" class="border border-gray-600 bg-gray-700 text-white rounded px-2 py-1 text-xs w-24">
				  <option value="dubbing">dubbing</option>
				  <option value="normal">normal</option>
				</select>
				<span class="text-sm text-gray-300">Lang</span>
				<select id="ts-lang" class="border border-gray-600 bg-gray-700 text-white rounded px-2 py-1 text-xs w-16">
				  <option value="id" ${st.lang==='id'?'selected':''}>ID</option>
				  <option value="en" ${st.lang==='en'?'selected':''}>EN</option>
				</select>
				<span class="text-sm text-gray-300">Engine</span>
				<select id="ts-engine" class="border border-gray-600 bg-gray-700 text-white rounded px-2 py-1 text-xs w-20">
				  <option value="llm" ${st.engine==='llm'?'selected':''}>LLM</option>
				  <option value="nllb" ${st.engine==='nllb'?'selected':''}>NLLB</option>
				  <option value="whisper" ${st.engine==='whisper'?'selected':''}>Whisper</option>
				</select>
				<span class="text-sm text-gray-300">Temp</span>
				<input id="ts-temp" type="number" step="0.1" min="0" max="2" value="${st.temperature}"
					   class="border border-gray-600 bg-gray-700 text-white rounded px-1 py-1 w-10 text-xs" />
				<span class="text-sm text-gray-300">top-p</span>
				<input id="ts-topp" type="number" step="0.05" min="0" max="1" value="${st.top_p}"
					   class="border border-gray-600 bg-gray-700 text-white rounded px-1 py-1 w-10 text-xs" />
			  </div>
			</div>

			<!-- Main content area - 1 BARIS SAJA -->
			<div class="flex gap-3 h-[65vh]">
			  <!-- LEFT: Subtitle Viewer -->
			  <div class="flex-1 flex flex-col">
				<div class="flex items-center gap-3 mb-2">
				  <div class="text-base font-semibold text-white">Subtitles</div>
				  <div class="flex items-center gap-2">
					<span class="text-sm text-gray-300">Search:</span>
					<input id="ts-search" type="text" placeholder="..." class="border border-gray-600 bg-gray-700 text-white rounded px-2 py-1 w-32 text-sm" />
				  </div>
				  <div class="ml-auto flex items-center gap-2">
					<button id="ts-prev" class="btn btn-slate text-xs px-2 py-1">‹</button>
					<button id="ts-nextpage" class="btn btn-slate text-xs px-2 py-1">›</button>
					<span class="text-xs text-gray-300">Page</span>
					<input id="ts-page" type="number" min="1" value="${st.page}" class="border border-gray-600 bg-gray-700 text-white rounded px-1 py-1 w-10 text-xs" />
					<span class="text-xs text-gray-300">/<span id="ts-pages">1</span></span>
					<select id="ts-size" class="border border-gray-600 bg-gray-700 text-white rounded px-1 py-1 text-xs w-16">
					  ${[100,500,1000,3000,7000].map(n=>`<option value="${n}" ${st.size===n?'selected':''}>${n}</option>`).join('')}
					</select>
					<label class="inline-flex items-center gap-1">
					  <input id="ts-follow" type="checkbox" ${st.follow?'checked':''} class="w-3 h-3 text-blue-500 bg-gray-600 border-gray-500 rounded focus:ring-blue-400" />
					  <span class="text-xs text-gray-300">Follow</span>
					</label>
				  </div>
				</div>
				<div id="ts-table" class="flex-1 border border-gray-600 rounded overflow-auto bg-gray-900"></div>
			  </div>

			  <!-- RIGHT: Messages -->
			  <div class="w-1/3 flex flex-col">
				<div class="text-base font-semibold text-white mb-2">Results</div>
				<pre id="ts-messages" class="flex-1 border border-gray-600 rounded p-2 overflow-auto bg-gray-900 text-green-200 text-xs"></pre>
			  </div>
			</div>
		  </div>
		</div>
	  `;
	  tabContent.dataset.inited = '1';

	  this.setupTranslateEvents();
	}

// ==================== TRANSLATE: EVENT BINDING (always re-bind after render) ====================
setupTranslateEvents() {
  if (!this.translate) this.initTranslateState();
  const $ = (id) => document.getElementById(id);

  // -- input config (sinkron ke state) --
  const bindVal = (id, key, toNum=false) => {
    const el = $(id); if (!el) return;
    el.addEventListener('input', () => { this.translate[key] = toNum ? Number(el.value) : el.value; });
  };
  const bindChk = (id, key) => {
    const el = $(id); if (!el) return;
    el.addEventListener('change', () => { this.translate[key] = !!el.checked; });
  };

  bindVal('ts-api',     'apiKey');
  bindVal('ts-batch',   'batch', true);
  bindVal('ts-workers', 'workers', true);
  bindVal('ts-temp',    'temperature', true);
  bindVal('ts-topp',    'top_p', true);
  bindVal('ts-lang',    'lang');
  bindVal('ts-engine',  'engine');
  bindVal('ts-style',   'style');
  bindChk('ts-autosave','autosave');

  // -- search & paging --
  $('ts-search')?.addEventListener('input', () => { this.translate.search = $('ts-search').value.trim(); this.filterAndRenderSubs(); });
  $('ts-size')?.addEventListener('change', () => { this.translate.size = Number($('ts-size').value) || 7000; this.translate.page = 1; this.renderSubsPage(); });
  $('ts-page')?.addEventListener('change', () => { this.translate.page = Math.max(1, Number($('ts-page').value || 1)); this.renderSubsPage(); });
  $('ts-prev')?.addEventListener('click',  () => { this.translate.page = Math.max(1, this.translate.page - 1); this.renderSubsPage(); });
  $('ts-nextpage')?.addEventListener('click', () => { this.translate.page = Math.min(this.totalPages || 1, this.translate.page + 1); this.renderSubsPage(); });
  $('ts-follow')?.addEventListener('change', () => { this.translate.follow = $('ts-follow').checked; });

  // -- file/save/export/clear/next-tab --
  $('ts-load')?.addEventListener('click',  () => this.promptLoadSRT());
  $('ts-save')?.addEventListener('click',  () => this.translateSave());
  $('ts-export')?.addEventListener('click',() => this.translateExport());
  $('ts-clear')?.addEventListener('click', () => this.translateClear());
  $('ts-next')?.addEventListener('click',  () => this.loadTab('editing'));

  // -- STREAMING actions (no space before ?.) --
  $('ts-start')?.addEventListener('click',  () => this.tsStartStream('start'));
  $('ts-resume')?.addEventListener('click', () => this.tsStartStream('resume'));
  $('ts-fill')?.addEventListener('click',   () => this.tsStartStream('fill'));
  $('ts-stop')?.addEventListener('click',   () => this.tsStopStream());

  // render awal
  this.renderSubsPage();
  this.renderMessages();
}

	/* ==============================
	 * LOAD SRT (auto/manual)
	 * ============================== */
	async promptLoadSRT() {
	  const setText = (text, mode) => {
		this.translate.mode = mode;
		this.translate.srtText = text;
		const subs = this.parseSRT(text);
		this.translate.originalSubs = subs.map(s => ({...s})); // keep original
		// working copy dengan placeholder trans=''
		this.translate.subtitles = subs.map(s => ({...s, trans: ''}));
		this.filterAndRenderSubs();
		this.updateTranslateProgress();
		this.showNotification(mode==='auto'?'SRT loaded from session':'Loaded SRT file','success');
	  };

	  if (this.currentSessionId) {
		try {
		  const res = await fetch(`/api/session/${this.currentSessionId}/srt?prefer=original`, { method: 'GET' });
		  let text = await res.text();
		  if (!res.ok) throw new Error(text);
		  try { const obj=JSON.parse(text); if (obj && obj.srt) text = obj.srt; } catch {}
		  return setText(text,'auto');
		} catch (e) {
		  this.showNotification('Backend SRT tidak tersedia, pilih file manual…','warning');
		}
	  }
	  // manual picker
	  const input = document.createElement('input'); input.type='file'; input.accept='.srt';
	  input.onchange = async ()=>{ const f=input.files?.[0]; if(!f) return; setText(await f.text(),'manual'); };
	  input.click();
	}

	/* ==============================
	 * TRANSLATE HELPER FUNCTIONS
	 * ============================== */

	_getIndicesForStart(overwrite = false) {
	  const st = this.translate;
	  if (!st?.subtitles) return [];
	  if (overwrite) return st.subtitles.map(s => s.index);
	  // default: hanya yang kosong
	  return st.subtitles.filter(s => !(s.trans || '').trim()).map(s => s.index);
	}

	_getIndicesForResume() {
	  const st = this.translate;
	  if (!st?.subtitles) return [];
	  const cp = st.checkpoint || 0;
	  return st.subtitles
		.filter(s => s.index > cp && !(s.trans || '').trim())
		.map(s => s.index);
	}

	_getIndicesForFillMissing() {
	  const st = this.translate;
	  if (!st?.subtitles) return [];
	  return st.subtitles.filter(s => !(s.trans || '').trim()).map(s => s.index);
	}
	
	/* ==============================
	 * RUN TRANSLATE (ALL / MISSING)
	 * ============================== */
	async translateStartAll(){ return this._translateGeneric(false); }
	async translateMissing(){  return this._translateGeneric(true); }

	// === TAB 2: jalankan translate dengan mode Start / Resume / Fill Missing ===
	async _translateGeneric() {
	  // --- state awal & helper kecil ---
	  if (!this.translate) this.initTranslateState();
	  const st = this.translate;

	  const $ = (id) => document.getElementById(id);
	  const btnStart  = $('ts-start');
	  const btnFill   = $('ts-fill');
	  const btnStop   = $('ts-stop');
	  const btnExport = $('ts-export');
	  const barEl     = $('ts-progressbar');
	  const labEl     = $('ts-progresslabel');

	  // Ambil parameter UI
	  const apiKeyEl = $('ts-api');
	  const batchEl  = $('ts-batch');
	  const workEl   = $('ts-workers');
	  const tempEl   = $('ts-temp');
	  const toppEl   = $('ts-topp');
	  const langSel  = $('ts-lang');
	  const styleSel = $('ts-style');

	  // Simpan konfigurasi
	  st.apiKey      = (apiKeyEl?.value || '').trim();
	  st.batch       = Number(batchEl?.value || 30);
	  st.workers     = Number(workEl?.value || 1);
	  st.temperature = Number(tempEl?.value || 0.1);
	  st.top_p       = Number(toppEl?.value || 0.3);
	  st.lang        = (langSel?.value || 'id').toLowerCase();
	  st.style       = (styleSel?.value || 'dubbing');

	  // Validasi sederhana
	  if (!this.currentSessionId && !st.srtText) {
		this.showNotification('Belum ada session & SRT; "Load" SRT dulu atau buat session di Tab 1.', 'error');
		return;
	  }
	  if (!st.apiKey) {
		this.showNotification('API key kosong. Masukkan API key dahulu.', 'error');
		return;
	  }
	  if (st.running) return;

	  // ----- mode eksekusi: start / resume / fill -----
	  // set oleh tombol sebelum memanggil _translateGeneric(); default 'start'
	  const mode = (st.runMode || 'start'); // 'start' | 'start-overwrite' | 'resume' | 'fill'

	  // Helper untuk memilih indeks target
	  const pickIndices = () => {
		if (!st?.subtitles) return [];
		const isEmpty = (s) => !((s.trans || '').trim());

		if (mode === 'resume') {
		  const cp = st.checkpoint || 0;
		  return st.subtitles.filter(s => s.index > cp && isEmpty(s)).map(s => s.index);
		}
		if (mode === 'fill') {
		  return st.subtitles.filter(isEmpty).map(s => s.index);
		}
		if (mode === 'start-overwrite') {
		  return st.subtitles.map(s => s.index);
		}
		// default 'start' aman: hanya baris kosong
		return st.subtitles.filter(isEmpty).map(s => s.index);
	  };

	  // Kunci tombol (Stop tetap aktif)
	  btnStart?.setAttribute('disabled','disabled');
	  btnFill?.setAttribute('disabled','disabled');
	  btnExport?.setAttribute('disabled','disabled');
	  btnStop?.removeAttribute('disabled');

	  st.running = true;

	  // Tampilkan overlay, tapi biarkan tombol stop tetap bisa diklik (kalau overlay kamu blok klik,
	  // kamu bisa set CSS overlay pointer-events:none; — tidak saya ubah di sini).
	  const overlay = this.showLoading('Translating SRT...');

	  // Pastikan SRT ORIGINAL sudah ada (untuk manual mode / untuk enforce prefer=original)
	  try {
		if (!st.srtText && this.currentSessionId) {
		  const resSrt = await fetch(`/api/session/${this.currentSessionId}/srt?prefer=original`);
		  if (!resSrt.ok) throw new Error('Gagal mengambil SRT session (prefer=original)');
		  st.srtText      = await resSrt.text();
		  st.originalSubs = this.parseSRT(st.srtText);
		  st.subtitles    = st.originalSubs.map(x => ({ ...x })); // reset grid kiri
		  this._renderTable?.();
		}
	  } catch (e) {
		this.hideLoading();
		st.running = false;
		btnStart?.removeAttribute('disabled');
		btnFill?.removeAttribute('disabled');
		btnExport?.removeAttribute('disabled');
		btnStop?.setAttribute('disabled','disabled');
		this.showNotification(`Load SRT gagal: ${e.message || e}`, 'error');
		return;
	  }

	  // Tentukan baris mana yang harus diterjemahkan
	  let indices = pickIndices();
	  indices = [...new Set(indices)].sort((a,b)=>a-b); // unik & urut

	  if (!indices.length) {
		// Sudah selesai semua atau tidak ada yang perlu diterjemahkan
		this.hideLoading();
		st.running = false;
		btnStart?.removeAttribute('disabled');
		btnFill?.removeAttribute('disabled');
		btnExport?.removeAttribute('disabled');
		btnStop?.setAttribute('disabled','disabled');
		this.showNotification('Tidak ada baris yang perlu diterjemahkan.', 'success');
		return;
	  }

	  // Build form payload
	  const form = new URLSearchParams({
		api_key: st.apiKey,
		target_lang: st.lang,
		engine: 'llm',
		temperature: String(st.temperature),
		top_p: String(st.top_p),
		batch: String(st.batch),
		workers: String(st.workers),
		timeout: '120',
		autosave: 'true',
		mode: st.style === 'dubbing' ? 'dubbing' : 'normal',
		srt_text: st.srtText,         // kirim sumber ORIGINAL
		prefer: 'original',
		only_indices: indices.join(','), // ← HANYA baris target
	  });

	  // Abortable
	  const ac = new AbortController();
	  st.abort = () => ac.abort();

	  // Progress bar helper
	  const total = indices.length;
	  let done = 0;
	  const updatePB = () => {
		const pct = Math.round((done / total) * 100);
		if (barEl) barEl.style.width = `${pct}%`;
		if (labEl) labEl.textContent = `${pct}% (${done}/${total})`;
	  };
	  updatePB();

	  try {
		// Pilih endpoint: pakai /api/translate untuk manual (tanpa session)
		const resp = await fetch(
		  this.currentSessionId
			? `/api/session/${this.currentSessionId}/translate`
			: `/api/translate`,
		  {
			method: 'POST',
			headers: { 'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8' },
			body: form,
			signal: ac.signal
		  }
		);

		const raw = await resp.text();
		if (!resp.ok) throw new Error(raw || `HTTP ${resp.status}`);

		// Normalisasi respons: {results: [...]} ATAU SRT utuh
		let results = null;
		let json = null;
		try { json = JSON.parse(raw); } catch {}
		if (json && Array.isArray(json.results)) {
		  results = json.results;
		} else {
		  const outSrt =
			json?.data?.translated_srt ||
			json?.translated_srt ||
			json?.srt ||
			raw;

		  // Parse SRT → list hasil
		  const parsed = this.parseSRT(outSrt); // [{index,start,end,text}]
		  results = parsed.map(p => {
			const idxNo = Number(p.index);
			const ts    = `${p.start} --> ${p.end}`;
			// cocokan ke sumber
			let src = st.subtitles.find(s => s.index === idxNo);
			if (!src) src = st.subtitles.find(s => (`${s.start} --> ${s.end}`) === ts);
			return {
			  index: idxNo,
			  timestamp: ts,
			  original_text: src ? (src.text || '') : '',
			  translation: (p.text || '').trim()
			};
		  });
		}
		if (!Array.isArray(results)) throw new Error('Tidak ada hasil terjemahan.');

		// Tulis hasil per-item (realtime di panel kanan & grid kiri)
		st.doneSet = st.doneSet || new Set();
		for (const item of results) {
		  // Cari index di grid
		  let idx = -1;
		  if (Number.isInteger(item.index)) {
			idx = st.subtitles.findIndex(s => s.index === item.index);
		  }
		  if (idx < 0 && item.timestamp) {
			idx = st.subtitles.findIndex(s => `${s.start} --> ${s.end}` === item.timestamp);
		  }
		  if (idx < 0 && item.original_text) {
			idx = st.subtitles.findIndex(s => (s.text || '') === item.original_text);
		  }

		  const trans = (item.translation ?? '').trim();
		  if (idx >= 0) {
			// hanya isi target; jangan menimpa yang sudah ada kecuali mode 'start-overwrite'
			const already = (st.subtitles[idx].trans || '').trim();
			const canWrite = (mode === 'start-overwrite') || !already;
			if (canWrite) {
			  st.subtitles[idx].trans = trans;
			  this._renderTableRow?.(idx);
			}

			// Update checkpoint & doneSet untuk Resume
			st.doneSet.add(st.subtitles[idx].index);
			st.checkpoint = Math.max(st.checkpoint || 0, st.subtitles[idx].index);
		  }

		  // kirim ke panel kanan (biar “jalan” per baris)
		  this.appendMessage?.({ event: 'translate:progress', payload: item });

		  done += 1;
		  updatePB();
		}

		this._lastTranslatedAt = Date.now();
		this.showNotification('Translate selesai.', 'success');

	  } catch (err) {
		this.appendMessage?.({ event: 'translate:error', payload: String(err) });
		this.showNotification(`Translate gagal: ${err.message || err}`, 'error');

	  } finally {
		st.running = false;
		this.hideLoading();
		// Pulihkan tombol
		btnStart?.removeAttribute('disabled');
		btnFill?.removeAttribute('disabled');
		btnExport?.removeAttribute('disabled');
		btnStop?.setAttribute('disabled','disabled');
	  }
	}

	// === Export hasil translate menjadi .srt (download) ===
	async _exportSRT() {
	  if (!this.translate) this.initTranslateState();
	  const st = this.translate;
	  if (!st.subtitles?.length) {
		this.showNotification('Tidak ada data subtitle untuk diekspor.', 'error');
		return;
	  }

	  const lines = [];
	  for (const it of st.subtitles) {
		const out = (it.trans ?? '').trim(); // biarkan kosong kalau belum diterjemahkan
		lines.push(String(it.index));
		lines.push(`${it.start} --> ${it.end}`);
		lines.push(out);
		lines.push(''); // pemisah
	  }
	  const srtOut = lines.join('\r\n');

	  const blob = new Blob([srtOut], { type: 'text/plain;charset=utf-8' });
	  const url  = URL.createObjectURL(blob);
	  const a    = document.createElement('a');
	  a.href     = url;
	  a.download = 'translated.srt';
	  document.body.appendChild(a);
	  a.click();
	  a.remove();
	  URL.revokeObjectURL(url);

	  this.showNotification('Export SRT selesai (download).', 'success');
	}

	// --- PATCH: helper untuk refresh baris saat streaming progress ---
	// Versi sederhana: render ulang halaman aktif (aman & cepat dikerjakan).
	_renderTableRow(idx) {
	  // Kalau nanti mau super-presisi per baris, kita bisa kasih id pada setiap row
	  // lalu update innerHTML row tsb. Untuk sekarang cukup render ulang halaman.
	  if (typeof this.renderSubsPage === 'function') {
		this.renderSubsPage();
	  }
	  if (typeof this.updateTranslateProgress === 'function') {
		this.updateTranslateProgress();
	  }
	  if (typeof this.renderMessages === 'function') {
		this.renderMessages();
	  }
	}

	// (opsional) kecilkan jejak UI refresh total
	_renderTable() { 
	  // alias agar panggilan lama ke _renderTable tidak error
	  if (typeof this.renderSubsPage === 'function') this.renderSubsPage();
	}


	/* ==============================
	 * MERGE: translated SRT -> subtitles[].trans
	 * ============================== */
	mergeTranslatedSRT(translatedText, partial=false) {
	  const parsed = this.parseSRT(translatedText);   // sama struktur: index/start/end/text
	  // buat map index -> text
	  const tmap = new Map(parsed.map(p => [Number(p.index), (p.text||'').trim()]));
	  // merge
	  this.translate.subtitles = this.translate.subtitles.map(s => {
		const t = tmap.get(Number(s.index));
		if (t && (!partial || !s.trans)) {
		  return {...s, trans: t};
		}
		return s;
	  });
	  this.filterAndRenderSubs();
	  this.renderMessages(); // refresh JSON kanan
	}

	/* ==============================
	 * PROGRESS BAR
	 * ============================== */
updateTranslateProgress(done, total) {
  const pct = total ? Math.round((done / total) * 100) : 0;
  const bar = document.getElementById('ts-progressbar');
  const lab = document.getElementById('ts-progresslabel');
  if (bar) bar.style.width = `${pct}%`;
  if (lab) lab.textContent = `${pct}% (${done}/${total})`;
}

// Fokus/scroll ke baris index tertentu
scrollToTsRow(index) {
  const row = document.getElementById(`ts-row-${index}`);
  if (row) row.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

setRowTranslation(index, text) {
  const inp = document.querySelector(`input.ts-trans[data-idx="${index}"]`);
  if (inp) {
    inp.value = text || '';
    inp.classList.add('bg-green-900/40'); // efek kecil biar kelihatan update
    setTimeout(()=>inp.classList.remove('bg-green-900/40'), 300);
  }
  // sinkronkan ke state
  const row = this.translate?.subtitles?.find(r => r.index === index);
  if (row) row.trans = text || '';
}


	/* ==============================
	 * SAVE / EXPORT / STOP / CLEAR
	 * ============================== */
	translateSave(){
	  // rakit SRT dari state (pakai trans kalau ada, fallback ke original)
	  const srt = this.translate.subtitles.map(s => {
		const txt = (s.trans && s.trans.trim()) ? s.trans : (s.text || "");
		return `${s.index}\n${s.start} --> ${s.end}\n${txt}\n`;
	  }).join('\n');

	  // Kalau ada session → simpan ke workdir lewat backend
	  if (this.currentSessionId) {
		const body = new URLSearchParams({
		  srt_text: srt,
		  filename: "translated_latest.srt"
		});
		this.showLoading("Saving to workspace...");
		fetch(`/api/session/${this.currentSessionId}/save-srt`, {
		  method: "POST",
		  headers: {"Content-Type":"application/x-www-form-urlencoded"},
		  body
		})
		.then(async (res) => {
		  const txt = await res.text();
		  if (!res.ok) throw new Error(txt);
		  const data = JSON.parse(txt);
		  this.showNotification(`Saved: ${data.path}`, "success");
		})
		.catch(err => {
		  this.showNotification(`Save failed: ${err}`, "error");
		})
		.finally(()=> this.hideLoading());
		return;
	  }

	  // Tanpa session → fallback download lokal
	  const blob = new Blob([srt], {type:'text/plain;charset=utf-8'});
	  const a = document.createElement('a');
	  a.href = URL.createObjectURL(blob);
	  a.download = 'translated.srt';
	  a.click();
	  URL.revokeObjectURL(a.href);
	}
	
	translateExport() {
	  // rakit SRT dari state: pakai terjemahan jika ada, fallback ke original
	  const srt = (this.translate.subtitles || []).map(s => {
		const txt = (s.trans && s.trans.trim()) ? s.trans : (s.text || "");
		return `${s.index}\n${s.start} --> ${s.end}\n${txt}\n`;
	  }).join('\n');

	  if (this.currentSessionId) {
		// simpan ke workspace session
		const body = new URLSearchParams({
		  srt_text: srt,
		  filename: "translated_export.srt"
		});
		this.showNotification("Saving SRT to workspace…", "info");
		fetch(`/api/session/${this.currentSessionId}/save-srt`, {
		  method: "POST",
		  headers: { "Content-Type": "application/x-www-form-urlencoded" },
		  body
		})
		.then(async r => {
		  const t = await r.text();
		  if (!r.ok) throw new Error(t);
		  const data = JSON.parse(t);
		  this.showNotification(`Exported: ${data.path}`, "success");
		})
		.catch(err => this.showNotification(`Export failed: ${err}`, "error"));
		return;
	  }

	  // tanpa session → download lokal
	  const blob = new Blob([srt], { type: "text/plain;charset=utf-8" });
	  const a = document.createElement("a");
	  a.href = URL.createObjectURL(blob);
	  a.download = "translated.srt";
	  a.click();
	  URL.revokeObjectURL(a.href);
	}
	
	translateStop() {
	  try { if (this.translate?.abort) this.translate.abort(); } catch (e) {}
	}
	translateClear(){ this.initTranslateState(); this.loadTab('translate'); }

	/* ==============================
	 * VIEWER & MESSAGES
	 * ============================== */
	filterAndRenderSubs(){
	  const q=(this.translate.search||'').toLowerCase();
	  const src=this.translate.subtitles||[];
	  this.translate.filtered = q ? src.filter(x=> ((x.text||'') + ' ' + (x.trans||'')).toLowerCase().includes(q)) : src.slice();
	  this.translate.page=1; this.renderSubsPage();
	}
	
// Tambahkan method ini di dalam class DracinApp
buildSrtFromState() {
  // gunakan this.translate.subtitles (index/start/end/text)
  const rows = this.translate.subtitles || [];
  return rows.map(r => `${r.index}\n${r.start} --> ${r.end}\n${(r.text||'')}\n`).join('\n');
}

// ==================== HELPERS ====================
_tsMissingIndices() {
  // baris yang belum punya terjemahan (kosong)
  return this.translate.subtitles
    .filter(x => !x.trans || !String(x.trans).trim())
    .map(x => x.index);
}

_tsResumeFromFirstGap(missingList) {
  if (!missingList.length) return [];
  const first = Math.min(...missingList);
  return this.translate.subtitles
    .filter(x => x.index >= first)
    .map(x => x.index);
}

_tsToggleButtons(disabled) {
  ['ts-start','ts-resume','ts-fill','ts-stop','ts-load','ts-save','ts-export','ts-clear']
    .forEach(id => {
      const b = document.getElementById(id);
      if (b) b.disabled = !!disabled && id!=='ts-stop';
    });
}

_tsUpdateProgress(done, total) {
  const bar = document.getElementById('ts-progressbar');
  const label = document.getElementById('ts-progresslabel');
  const pct = (!total ? 0 : Math.round(100*done/total));
  if (bar)   bar.style.width = `${pct}%`;
  if (label) label.textContent = `${pct}% (${done}/${total||0})`;
}

// ==================== TRANSLATE: STREAM START/RESUME/FILL ====================
// === ADD/REPLACE: jalankan terjemahan STREAM (start/resume/fill) ===
async tsStartStream(runMode='start') {
  if (!this.translate) this.initTranslateState();
  const st = this.translate;

  if (st.running) { this.showNotification('Translate is running…', 'info'); return; }
  if (!st.apiKey)  { this.showNotification('API key kosong.', 'error'); return; }

  // Tentukan indeks target
  const allIdx  = (st.subtitles || []).map(x => x.index);
  const missing = (st.subtitles || []).filter(x => !(x.trans||'').trim()).map(x => x.index);
  let only = [];
  if      (runMode === 'fill')   only = missing;
  else if (runMode === 'resume') {
    if (!missing.length) { this.showNotification('Nothing to resume.', 'info'); return; }
    const firstGap = Math.min(...missing);
    only = st.subtitles.filter(x => x.index >= firstGap && !(x.trans||'').trim()).map(x => x.index);
  } else { // start
    only = allIdx;
  }
  if (!only.length) { this.showNotification('Tidak ada baris yang perlu diterjemahkan.', 'success'); return; }

  const hasSession = !!this.currentSessionId;
  const url = hasSession
    ? `/api/session/${this.currentSessionId}/translate/stream`
    : `/api/translate/stream`;

  // FormData untuk STREAM (pakai batch & typing delay)
  const fd = new FormData();
  fd.append('api_key', st.apiKey);
  fd.append('target_lang', st.lang || 'id');
  fd.append('engine', st.engine || 'llm');
  fd.append('temperature', String(st.temperature ?? 0.1));
  fd.append('top_p', String(st.top_p ?? 0.3));
  fd.append('workers', String(st.workers || 1));
  fd.append('timeout', String(st.timeout || 120));
  fd.append('mode', st.style || 'dubbing');
  fd.append('only_indices', only.join(','));
  fd.append('batch', String(st.batch || 30));                 // penting
  fd.append('typing_delay_ms', String(st.typingDelayMs));

  if (hasSession) {
    fd.append('prefer', 'original');                // ambil SRT original di server
    fd.append('autosave', st.autosave ? 'true':'false');
  } else {
    // manual mode: kirim SRT yang ada di state
    const src = (st.originalSubs && st.originalSubs.length)
      ? st.originalSubs
      : st.subtitles;
    const srtText = (src || []).map(it => `${it.index}\n${it.start} --> ${it.end}\n${it.text||''}\n`).join('\n');
    if (!srtText.trim()) { this.showNotification('SRT belum dimuat. Klik Load dulu.', 'error'); return; }
    fd.append('srt_text', srtText);
  }

  // Lock UI + progress
  st.running = true;
  ['ts-start','ts-resume','ts-fill','ts-load','ts-save','ts-export','ts-clear','ts-next'].forEach(id=>{
    const b=document.getElementById(id); if (b && id!=='ts-stop') b.disabled = true;
  });
  const total = only.length;
  let done = 0;
  const upd = () => {
    const bar = document.getElementById('ts-progressbar');
    const lab = document.getElementById('ts-progresslabel');
    const pct = Math.round(done*100/total);
    if (bar) bar.style.width = `${pct}%`;
    if (lab) lab.textContent = `${pct}% (${done}/${total})`;
  };
  upd();

  // siapkan abort
  const controller = new AbortController();
  st.abort = controller;

  try {
    const res = await fetch(url, { method:'POST', body: fd, signal: controller.signal });
    if (!res.ok || !res.body) {
      const msg = await res.text().catch(()=>`${res.status} ${res.statusText}`);
      throw new Error(`Translate failed: ${msg}`);
    }

    // NDJSON reader
    const reader  = res.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';

    // init progress dari server (opsional)
    this.appendMessage({ type:'begin', url, batch: st.batch, workers: st.workers });

    while (true) {
      const {value, done:drained} = await reader.read();
      if (drained) break;
      buffer += decoder.decode(value, {stream:true});

      let nl;
      while ((nl = buffer.indexOf('\n')) >= 0) {
        const line = buffer.slice(0, nl).trim();
        buffer = buffer.slice(nl+1);
        if (!line) continue;

        let evt;
        try { evt = JSON.parse(line); } catch { continue; }

        // tampilkan di panel kanan (debug/result)
        this.appendMessage(evt);

		if (evt.type === 'result' || evt.type === 'progress') {
		  this._enqueueStreamEvent(evt);
		  continue;
		}

		// event lain (begin/end/error) → tampilkan di panel kanan
		this.appendMessage(evt);
      }
    }

    this.showNotification('Translate finished', 'success');
  } catch (err) {
    if (err?.name === 'AbortError') this.showNotification('Translate stopped', 'info');
    else this.showNotification(`Translate failed: ${err?.message || err}`, 'error');
  } finally {
    st.running = false;
    st.abort   = null;
    // unlock UI
    ['ts-start','ts-resume','ts-fill','ts-load','ts-save','ts-export','ts-clear','ts-next'].forEach(id=>{
      const b=document.getElementById(id); if (b) b.disabled = false;
    });
    const bar = document.getElementById('ts-progressbar');
    const lab = document.getElementById('ts-progresslabel');
    if (bar) bar.style.width = '0%';
    if (lab) lab.textContent = `0% (0/0)`;
  }
}


// Bangun SRT dari originalSubs (tanpa terjemahan) – untuk mode manual
buildSrtFromOriginal(items = []) {
  return (items || []).map(it => {
    return `${it.index}\n${it.start} --> ${it.end}\n${it.text || ''}\n`;
  }).join('\n');
}

// === HANYA yang kosong ===
getMissingIndices() {
  return (this.translate?.subtitles || [])
    .filter(r => !(r.trans || '').trim())
    .map(r => r.index);
}

// === Resume: dari first kosong sampai akhir, yang kosong saja ===
getResumeIndices() {
  const arr = this.translate?.subtitles || [];
  const pos = arr.findIndex(r => !(r.trans || '').trim());
  if (pos < 0) return [];
  return arr.slice(pos).filter(r => !(r.trans || '').trim()).map(r => r.index);
}

// Panggil di setupTranslateEvents() untuk tombol Fill/Resume:
tsFillStream() {
  const idxs = this.getMissingIndices();
  if (!idxs.length) {
    this.showNotification('Nothing to fill. All translated.', 'info');
    return;
  }
  this.tsStartStream({ onlyIndices: idxs });
}

tsResumeStream() {
  if (!this.translate) this.initTranslateState();
  const rows = this.translate.subtitles || [];
  const cp = this.translate.checkpoint || 0;
  const only = rows.filter(r => r.index > cp && !(r.trans||'').trim()).map(r => r.index);
  return this.tsStartStream({ onlyIndices: only });
}

tsFillStream() {
  if (!this.translate) this.initTranslateState();
  const rows = this.translate.subtitles || [];
  const only = rows.filter(r => !(r.trans||'').trim()).map(r => r.index);
  return this.tsStartStream({ onlyIndices: only });
}

// ==================== TRANSLATE: STOP ====================
tsStopStream() {
  const st = this.translate || {};
  if (st.abort) { try { st.abort.abort(); } catch {} }
}


	// ===== Helper progress kecil di header =====
	tsUpdateProgress(done, total) {
	  this.translate.totalDone = done;
	  this.translate.totalAll  = total;
	  const pct = total ? Math.floor((done/total)*100) : 0;
	  const bar = document.getElementById('ts-progressbar');
	  const lbl = document.getElementById('ts-progresslabel');
	  if (bar) bar.style.width = `${pct}%`;
	  if (lbl) lbl.textContent = `${pct}% (${done}/${total})`;
	}
	 
// Di dalam method renderSubsPage(), pastikan markup seperti ini:
/**
 * Render tabel subtitle TANPA paksa scroll ke bawah.
 * - keepPos: true = jaga posisi scroll sekarang (default).
 * - Kalau kamu sudah punya renderSubsPage(), ganti dengan ini.
 */
renderSubsPage(keepPos = true) {
  if (!this.translate) this.initTranslateState();
  const st = this.translate;
  const wrap = document.getElementById('ts-table');
  if (!wrap) return;

  // simpan posisi scroll supaya tidak loncat ke bawah sesudah re-render
  const prevScroll = keepPos ? wrap.scrollTop : 0;

  // filter/paging sesuai state-mu (sesuaikan kalau berbeda)
  const list = st.filtered?.length ? st.filtered : (st.subtitles || []);
  const page = Math.max(1, Number(st.page || 1));
  const size = Number(st.size || 100);
  const start = (page - 1) * size;
  const pageRows = list.slice(start, start + size);

  // builder baris; PASTIKAN id="ts-row-{{index}}" dan input .ts-trans
const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({
  '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'
}[c]));

wrap.innerHTML = `
  <table class="w-full text-sm text-gray-200">
    <thead>
      <tr class="bg-gray-800 border-b border-gray-600 sticky top-0 text-sm font-semibold text-gray-300">
        <th class="px-3 py-2 w-10 text-left">#</th>
        <th class="px-3 py-2 w-20 text-left">Start</th>
        <th class="px-3 py-2 w-20 text-left">End</th>
        <th class="px-3 py-2 text-left">Original</th>
        <th class="px-3 py-2 w-[28rem] text-left">Translation</th>
      </tr>
    </thead>
    <tbody>
      ${pageRows.map(r => `
        <tr id="ts-row-${r.index}" class="border-b border-gray-700 hover:bg-gray-800">
          <td class="px-3 py-2 text-gray-400 font-mono">${r.index}</td>
          <td class="px-3 py-2 text-gray-300 font-mono text-xs">${esc(r.start)}</td>
          <td class="px-3 py-2 text-gray-300 font-mono text-xs">${esc(r.end)}</td>
          <td class="px-3 py-2 text-gray-300 break-words">${esc(r.text)}</td>
          <td class="px-3 py-2">
            <input class="ts-trans w-full bg-gray-800 border border-gray-700 rounded px-2 py-1 text-green-400 break-words font-medium" 
                   value="${esc(r.trans)}" data-idx="${r.index}">
          </td>
        </tr>
      `).join('')}
    </tbody>
  </table>
`;

  // restore posisi scroll (agar tidak turun ke bawah)
  if (keepPos) wrap.scrollTop = prevScroll;

  // sinkronisasi perubahan manual di input ke state (opsional)
  wrap.querySelectorAll('.ts-trans').forEach(inp => {
    inp.addEventListener('input', e => {
      const idx = Number(e.target.dataset.idx);
      const row = st.subtitles?.find(x => x.index === idx);
      if (row) row.trans = e.target.value;
    });
  });
}

/* ================== /PATCH: follow ke baris hasil terjemahan ================= */
	
	buildMessagesArray(){
	  return (this.translate.subtitles||[]).map(s=>({
		index: s.index,
		timestamp: `${s.start} --> ${s.end}`,
		original_text: s.text || "",
		translation: s.trans || ""
	  }));
	}

// ==================== RENDER LOG (batasi agar tidak “massal”) ====================
renderMessages(forceScroll = false) {
  const el = document.getElementById('ts-messages');
  if (!el || !this.translate) return;
  const msgs = this.translate.messages.slice(-200);
  el.textContent = JSON.stringify(msgs, null, 2);
  
  if (forceScroll || this.translate.followResults) {
    el.scrollTop = el.scrollHeight;
  }
}

/* ===========================
 *  TAB 3 — EDITING (FULL BLOCK)
 *  letakkan semua method ini DI DALAM class DracinApp
 * =========================== */
/* ===========================
 *  TAB 3 — EDITING (FULL BLOCK)
 *  (Semua method di bawah ini berada DI DALAM class DracinApp)
 * =========================== */
// === ADD: init Bulk UI sekali setelah _edLoad() ===
_edInitBulkUI() {
  const ed = this.editing || (this.editing = {});
  ed.selected  = ed.selected  || new Set();
  ed.bulkScope = ed.bulkScope || 'selected';

  const $ = (id) => document.getElementById(id);

  $('#ed-select-shown')?.addEventListener('click', () => this._edSelectShown(true));
  $('#ed-clear-selection')?.addEventListener('click', () => this._edSelectShown(false));
  $('#ed-invert-selection')?.addEventListener('click', () => this._edInvertSelection());

  document.querySelectorAll('input[name="ed-bulk-scope"]').forEach(radio => {
    radio.addEventListener('change', (e) => { ed.bulkScope = e.target.value; });
  });

	// master checkbox: select/clear all shown
	const m = document.getElementById('ed-master-chk');
	if (m && !m._bound) {
	  m._bound = true;
	  m.addEventListener('change', (e) => this._edMasterToggle(e.target.checked));
	}

  // Gender bulk
  [['ed-bulk-g-m','male'], ['ed-bulk-g-f','female'], ['ed-bulk-g-u','unknown']].forEach(([id,g]) => {
    $(id)?.addEventListener('click', () => this._edBulkSetGender(g));
  });

  // Speaker bulk
  $('#ed-bulk-set-speaker')?.addEventListener('click', () => {
    const name = ($('#ed-bulk-speaker')?.value || '').trim();
    if (!name) return this.showNotification('Isi speaker dulu.', 'warning');
    this._edBulkSetSpeaker(name);
  });
  $('#ed-bulk-clear-speaker')?.addEventListener('click', () => this._edBulkSetSpeaker(''));

  // Replace speaker (find/replace)
  $('#ed-bulk-replace')?.addEventListener('click', () => {
    const f = ($('#ed-bulk-find')?.value || '');
    const r = ($('#ed-bulk-repl')?.value || '');
    this._edBulkReplaceSpeaker(f, r);
  });

  this._edUpdateSelCount();
}

// Toggle master: pilih/bersihkan semua yang sedang tampil (shown)
_edMasterToggle(checked) {
  const ed = this.editing || (this.editing = {});
  ed.selected = ed.selected || new Set();
  const shown = (ed.filtered || []).map(r => r.index);
  for (const i of shown) checked ? ed.selected.add(i) : ed.selected.delete(i);
  this._edUpdateSelCount?.();
  this._edUpdateMasterChk?.();
  this._edRenderList?.(); // supaya checkbox di baris ikut sinkron
}

// Update status master (checked / indeterminate)
_edUpdateMasterChk() {
  const ed = this.editing || {};
  const master = document.getElementById('ed-master-chk');
  if (!master) return;

  const shown = (ed.filtered || []).map(r => r.index);
  let nSel = 0;
  for (const i of shown) if (ed.selected?.has(i)) nSel++;

  master.indeterminate = shown.length > 0 && nSel > 0 && nSel < shown.length;
  master.checked = shown.length > 0 && nSel === shown.length;
}

// === ADD: util untuk ambil target index sesuai scope ===
_edTargetIndices() {
  const ed = this.editing || {};
  const onlyEmpty = document.getElementById('ed-only-empty')?.checked;
  let idxs = [];

  if ((ed.bulkScope || 'selected') === 'shown') {
    idxs = (ed.filtered || []).map(r => r.index);
  } else {
    idxs = Array.from(ed.selected || []);
  }

  if (onlyEmpty) {
    const rows = ed.rows || [];
    idxs = idxs.filter(i => {
      const r = rows.find(x => x.index === i);
      return !((r?.speaker || '').trim());
    });
  }
  return idxs;
}

// === ADD: bulk set gender ===
_edBulkSetGender(g) {
  const ed = this.editing || {};
  const idxs = this._edTargetIndices();
  if (!idxs.length) return this.showNotification('Tidak ada baris terpilih/tersaring.', 'info');

  const patches = [];
  (ed.rows || []).forEach(r => {
    if (!idxs.includes(r.index)) return;
    const before = r.gender || 'unknown';
    if (before !== g) {
      patches.push({ idx:r.index, key:'gender', before, after:g });
      r.gender = g;
    }
  });

  this._edFilterRender();
  this._edRebuildVttSoon?.();
  this._edUpdateSelCount?.();
  if (patches.length) this._edPushHistory(`Bulk gender → ${g}`, patches);
}

// === ADD: bulk set / clear speaker ===
_edBulkSetSpeaker(name) {
  const ed   = this.editing || {};
  const idxs = this._edTargetIndicesSelected();  // langkah 3
  if (!idxs.length) {
    this.showNotification?.('Pilih baris dulu (pakai Check all kalau mau semuanya).', 'info');
    return;
  }

  const val = (name || '').trim() || null;

  // (opsional) untuk Undo/Redo kalau kamu sudah aktifkan modul history
  const patches = [];

  (ed.rows || []).forEach(r => {
    if (!idxs.includes(r.index)) return;
    const before = r.speaker || null;
    if (before !== val) {
      r.speaker = val;
      patches.push?.({ idx:r.index, key:'speaker', before, after:val });
    }
  });

  // refresh daftar speaker (untuk filter), lalu re-render list
  ed.speakers = [...new Set((ed.rows||[])
                  .map(x => (x.speaker||'').trim())
                  .filter(Boolean))].sort();
  this._edPopulateSpeakerFilter?.(ed.speakers);

  // re-render supaya warna input speaker ikut update
  this._edFilterRender?.();

  // sync UI kecil
  this._edUpdateSelCount?.();
  this._edUpdateMasterChk?.();

  // (opsional) catat history kalau ada
  if (patches.length && this._edPushHistory) {
    this._edPushHistory(`Bulk speaker → ${val||'(clear)'}`, patches);
  }

  this.showNotification?.(`Speaker di-set untuk ${idxs.length} baris.`, 'success');
}


// === ADD: bulk find/replace speaker (simple replace, case-sensitive) ===
_edBulkReplaceSpeaker(find, repl) {
  const ed = this.editing || {};
  const idxs = this._edTargetIndices();
  if (!idxs.length) return this.showNotification('Tidak ada baris terpilih/tersaring.', 'info');

  const patches = [];
  (ed.rows || []).forEach(r => {
    if (!idxs.includes(r.index)) return;
    const before = r.speaker || '';
    const after  = (before.replaceAll(find, repl)).trim() || null;
    if (before !== after) {
      patches.push({ idx:r.index, key:'speaker', before, after });
      r.speaker = after;
    }
  });

  ed.speakers = [...new Set((ed.rows||[]).map(r => (r.speaker||'').trim()).filter(Boolean))].sort();
  this._edPopulateSpeakerFilter(ed.speakers);
  this._edFilterRender();
  this._edUpdateSelCount?.();
  if (patches.length) this._edPushHistory(`Replace speaker "${find}"→"${repl}"`, patches);
}

// === ADD: helpers selection ===
_edSelectShown(checked) {
  const ed = this.editing || {};
  ed.selected = ed.selected || new Set();
  const shown = (ed.filtered || []).map(r => r.index);
  for (const i of shown) checked ? ed.selected.add(i) : ed.selected.delete(i);
  this._edRenderList?.();
  this._edUpdateSelCount();
  this._edUpdateMasterChk?.();
}

_edInvertSelection() {
  const ed = this.editing || {};
  ed.selected = ed.selected || new Set();
  const shown = (ed.filtered || []).map(r => r.index);
  for (const i of shown) ed.selected.has(i) ? ed.selected.delete(i) : ed.selected.add(i);
  this._edRenderList?.();
  this._edUpdateSelCount();
  this._edUpdateMasterChk?.();
}

_edUpdateSelCount() {
  const n = (this.editing?.selected?.size || 0);
  const el = document.getElementById('ed-sel-count');
  if (el) el.textContent = `${n} selected`;
}

// === GENDER COLORS (fixed) ===
// normalisasi nama agar konsisten (hindari "spk_01 " vs "SPK_01")
_edCanonSpeaker(name) {
  return String(name||'').trim().replace(/\s+/g,' ').toUpperCase();
}

// simpan/muat map warna di localStorage (per session)
_edLoadSpeakerColors() {
  const ed = this.editing || (this.editing = {});
  const sid = this.sessionId || ed.sessionId || 'default';
  try { ed._spkColor = JSON.parse(localStorage.getItem(`dracindub.spkcolors.${sid}`) || '{}'); }
  catch { ed._spkColor = {}; }
}
_edSaveSpeakerColors() {
  const ed = this.editing || (this.editing = {});
  const sid = this.sessionId || ed.sessionId || 'default';
  try { localStorage.setItem(`dracindub.spkcolors.${sid}`, JSON.stringify(ed._spkColor||{})); } catch {}
}

// warna gelap tajam (nyaman di dark UI)
// normalisasi nama agar konsisten (hindari "spk_01 " vs "SPK_01")
_edCanonSpeaker(name) {
  return String(name||'').trim().replace(/\s+/g,' ').toUpperCase();
}

// simpan/muat map warna di localStorage (per session)
_edLoadSpeakerColors() {
  const ed = this.editing || (this.editing = {});
  const sid = this.sessionId || ed.sessionId || 'default';
  try { ed._spkColor = JSON.parse(localStorage.getItem(`dracindub.spkcolors.${sid}`) || '{}'); }
  catch { ed._spkColor = {}; }
}
_edSaveSpeakerColors() {
  const ed = this.editing || (this.editing = {});
  const sid = this.sessionId || ed.sessionId || 'default';
  try { localStorage.setItem(`dracindub.spkcolors.${sid}`, JSON.stringify(ed._spkColor||{})); } catch {}
}

// warna gelap tajam (nyaman di dark UI)
_hash32(s){let h=2166136261>>>0;for(let i=0;i<(s||'').length;i++){h^=s.charCodeAt(i);h=Math.imul(h,16777619)>>>0;}return h>>>0;}
_speakerHueAvoid(h){const avoid=[[190,235],[315,350]];let hue=((h%360)+360)%360;const inR=(x,[a,b])=>a<=b?(x>=a&&x<=b):(x>=a||x<=b);let g=0;while(avoid.some(r=>inR(hue,r))&&g<20){hue=(hue+36)%360;g++;}return hue;}
_edEnsureSpeakerColor(name) {
  const ed = this.editing || (this.editing = {});
  ed._spkColor = ed._spkColor || {};
  const key = this._edCanonSpeaker(name);
  if (!key) return { solid: 'hsl(220 5% 35%)', bg: 'hsl(220 5% 35% / 0.18)' };
  if (!ed._spkColor[key]) {
    const hue = this._speakerHueAvoid(this._hash32(key));
    const solid = `hsl(${hue} 70% 38%)`;   // L rendah = nyaman
    const bg    = `hsl(${hue} 70% 38% / 0.18)`; // overlay tipis
    ed._spkColor[key] = { solid, bg };
    this._edSaveSpeakerColors();
  }
  return ed._spkColor[key];
}

// gender (gelap + overlay tipis)
_genderSolid(g){
  switch((g||'').toLowerCase()){
    case 'male':   return 'hsl(220 85% 46%)';
    case 'female': return 'hsl(340 76% 47%)';
    default:       return 'hsl(220  5% 55%)';
  }
}
// return "H,S%,L%" triplet → aman untuk hsl/hsla
_genderTriplet(g){
  switch ((g||'').toLowerCase()){
    case 'male':   return '220,85%,46%'; // blue-600
    case 'female': return '340,76%,47%'; // rose-600
    default:       return '220,5%,55%';  // gray-500
  }
}
_genderBorder(g){ return `hsl(${this._genderTriplet(g)})`; }            // solid
_genderBg(g){     return `hsla(${this._genderTriplet(g)},0.16)`; }      // overlay tipis


initEditingState() {
  this.editing = {
    sessionId: this.currentSessionId || "",
    rows: [],            // [{index,start,end,translation,gender,speaker,notes}]
    filtered: [],
    follow: true,
    genderFilter: "all",
    speakerFilter: "",
    search: "",
    shown: 0,
    total: 0,
    videoUrl: "",
    playingRowIndex: null,
    playTimer: null
  };
}

async loadEditingTab() {
  if (!this.editing) this.initEditingState();
  const st = this.editing;

  const tab = document.getElementById('editing');
  if (!tab) return;

tab.innerHTML = `
  <div id="tab-editing" class="bg-gray-800 rounded-lg p-4">
    <h2 class="text-xl font-bold mb-4 text-yellow-400">
      <i class="fas fa-edit mr-2"></i>Editing
    </h2>

    <!-- Session Controls -->
    <div class="flex flex-wrap items-center gap-2 mb-3">
      <input id="ed-session-input" class="bg-gray-700 border border-gray-600 text-white px-3 py-1 rounded w-64"
             placeholder="Paste / ketik Session ID…" />
      <button id="ed-load-session" class="btn btn-slate px-3 py-1">Load Session</button>

      <select id="ed-session-select" class="bg-gray-700 border border-gray-600 text-white px-2 py-1 rounded">
        <option value="">— pilih session —</option>
      </select>

      <span class="mx-2 text-gray-400">|</span>

      <select id="ed-gender" class="bg-gray-700 border border-gray-600 text-white px-2 py-1 rounded">
        <option value="all">All genders</option>
        <option value="male">male</option>
        <option value="female">female</option>
        <option value="unknown">unknown</option>
      </select>

      <select id="ed-speaker-select" class="bg-gray-700 border border-gray-600 text-white px-2 py-1 rounded">
        <option value="all">All speakers</option>
      </select>

      <input id="ed-search" class="bg-gray-700 border border-gray-600 text-white px-3 py-1 rounded w-52"
             placeholder="Search text…" />

      <label class="ml-2 inline-flex items-center gap-2 text-gray-300">
        <input id="ed-follow" type="checkbox" class="w-4 h-4" checked />
        Follow
      </label>
		<label class="ml-2 inline-flex items-center gap-2 text-gray-300">
		  <input id="ed-warn-only" type="checkbox" class="w-4 h-4" />
		  Warnings only
		</label>

      <div class="ml-auto flex items-center gap-2">
		<button id="ed-auto-sync"
		  class="bg-purple-600 hover:bg-purple-700 px-3 py-2 rounded text-white text-sm flex items-center">
		  <i class="fas fa-clock mr-1"></i>Auto Sync Timing
		</button>
        <button id="ed-save" class="btn btn-primary px-3 py-1">Save</button>
        <button id="ed-merge" class="btn bg-green-500 px-3 py-1" disabled>Gabung Subtitle</button>
        <button id="ed-exp-male" class="btn bg-pink-500 px-3 py-1">Export Male</button>
        <button id="ed-exp-female" class="btn bg-pink-500 px-3 py-1">Export Female</button>
        <button id="ed-exp-unk" class="btn bg-pink-500 px-3 py-1">Export Unknown</button>
        <button id="ed-exp-all" class="btn bg-pink-500 px-3 py-1">Gender All (zip)</button>
        <button id="ed-exp-speaker" class="btn bg-orange-500 px-3 py-1">Export Speaker Dipilih</button>
        <button id="ed-exp-speaker-zip" class="btn bg-orange-500 px-3 py-1">Semua Speakers (zip)</button>
        <button id="ed-exp-full" class="btn bg-red-500 px-3 py-1">Export Full SRT</button>
		  <button id="ed-reset"
      class="bg-red-500 hover:bg-red-600 px-3 py-2 rounded text-white text-sm flex items-center">
      <i class="fas fa-sync mr-1"></i>Reset Cache
  </button>
      </div>
    </div>

    <!-- Bulk Toolbar -->
    <div id="ed-bulkbar" class="p-3 rounded bg-gray-800/50 border border-gray-700">
      <div class="grid grid-cols-2 gap-4">
        <!-- Kolom Kiri - Selection & Actions -->
        <div class="space-y-2">
          <!-- Selection Row -->
          <div class="flex items-center gap-2">
            <button id="ed-select-shown" class="px-2 py-1 border rounded text-sm">Select shown</button>
            <button id="ed-clear-selection" class="px-2 py-1 border rounded text-sm">Clear</button>
            <button id="ed-invert-selection" class="px-2 py-1 border rounded text-sm">Invert</button>
            <span id="ed-sel-count" class="text-xs opacity-70 ml-1">0 selected</span>
          </div>

          <!-- Scope & Check Row -->
          <div class="flex items-center gap-2">
            <span class="text-sm text-gray-300">Apply to:</span>
            <label class="flex items-center gap-1">
              <input type="radio" name="ed-bulk-scope" value="selected" checked>
              <span class="text-sm">Selected</span>
            </label>
            
            <label class="flex items-center gap-1 ml-2">
              <input id="ed-master-chk" type="checkbox" class="w-4 h-4">
              <span class="text-sm">Check all</span>
            </label>
          </div>

          <!-- Gender & Speaker Row -->
          <div class="flex items-center gap-2">
            <span class="text-sm text-gray-300">Gender:</span>
            <button id="ed-bulk-g-m" class="btn bg-pink-500 px-2 py-1 border rounded text-sm">MALE</button>
            <button id="ed-bulk-g-f" class="btn bg-blue-500 px-2 py-1 border rounded text-sm">FEMALE</button>
            <button id="ed-bulk-g-u" class="px-2 py-1 border rounded text-sm">Unknown</button>

            <span class="text-sm text-gray-300 ml-2">Speaker:</span>
            <input id="ed-bulk-speaker" class="px-2 py-1 bg-gray-900 border border-gray-600 rounded w-28 text-sm"
                   placeholder="Nama speaker">
            <button id="ed-bulk-set-speaker" class="btn bg-blue-500 px-2 py-1 border rounded text-sm">
              Ubah
            </button>
			            <button id="ed-undo" class="btn bg-red-500 px-2 py-1 border rounded text-sm">Undo / Mundur</button>
            <button id="ed-redo" class="btn bg-green-500 px-2 py-1 border rounded text-sm">Redo / Maju</button>
          </div>

          <!-- Functions Row -->
        </div>

        <!-- Kolom Kanan - Mapping Settings -->
        <div class="space-y-2">
          <!-- ASSIGN_POLICY Row -->
          <div class="flex items-center gap-2">
            <label class="text-sm text-gray-300 whitespace-nowrap w-24">ASSIGN_POLICY:</label>
			
            <select id="assign-policy" class="bg-gray-600 border border-gray-500 rounded px-2 py-1 text-white text-sm flex-1">
			  <option value="overlap">overlap</option>
			  <option value="start">start</option>
			  <option value="end">end</option>
			  <option value="midpoint">midpoint</option>
			  <option value="majority_then_start">majority_then_start</option>
			  <option value="majority_then_midpoint">majority_then_midpoint</option>
			  <option value="adaptive" selected>adaptive</option>  <!-- default -->
            </select>
          </div>

          <!-- MAJORITY & SNAP_MS Row -->
          <div class="flex items-center gap-4">
            <div class="flex items-center gap-2 flex-1">
              <label class="text-sm text-gray-300 whitespace-nowrap w-16">MAJORITY:</label>
              <input id="majority" type="number" min="0" max="1" step="0.05" value="0.60"
                     class="bg-gray-600 border border-gray-500 rounded px-2 py-1 text-white text-sm w-20">
            </div>

            <div class="flex items-center gap-2 flex-1">
              <label class="text-sm text-gray-300 whitespace-nowrap w-16">SNAP_MS:</label>
              <input id="snap-ms" type="number" min="0" step="10" value="120"
                     class="bg-gray-600 border border-gray-500 rounded px-2 py-1 text-white text-sm w-20">
            </div>
          </div>

          <!-- Apply Button & Note -->
          <div class="flex items-center gap-2">
            <button id="ed-apply-mapping" class="btn bg-blue-600 px-3 py-1 rounded text-sm">
              Apply Mapping
            </button>
            <span class="text-xs text-gray-400">0 = matikan snapping</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Session Info -->
    <div id="ed-session-info" class="text-sm text-gray-400 my-3">No session loaded.</div>

    <!-- Main Content Grid -->
    <div class="grid grid-cols-[30%_1fr] gap-4 items-start">
      <!-- Video Panel -->
      <div class="sticky top-20 h-[calc(100vh-6rem)] bg-black rounded overflow-hidden">
        <video id="ed-video"
               class="w-full h-full object-contain bg-black aspect-[9/16]"
               controls preload="metadata" playsinline></video>
      </div>

      <!-- Subtitle List -->
      <div class="max-h-[calc(100vh-6rem)] overflow-auto flex flex-col">
        <div class="text-sm text-gray-300 mb-2">
          <span id="ed-counter">0 shown / 0 total</span>
        </div>
        <div id="ed-list" class="flex-1 space-y-2"></div>
      </div>
    </div>
  </div>
`;

  this._edBindEvents();
  await this._edPopulateSessionSelector();

  if (!st.sessionId && this.currentSessionId) st.sessionId = this.currentSessionId;
  if (st.sessionId) {
    const inp = document.getElementById('ed-session-input');
    if (inp) inp.value = st.sessionId;
    await this._edLoad(st.sessionId);
  }
  this._edInitBulkSimple?.();
}

_edUpdateMergeBtn() {
  const btn = document.getElementById('ed-merge');
  const ed = this.editing || {};
  const n = (ed && ed.selected) ? ed.selected.size : 0; // aman tanpa optional chaining
  if (btn) btn.disabled = (n < 2);
}

_edSecToSrt(sec) {
  sec = Math.max(0, Number(sec) || 0);
  const h  = Math.floor(sec / 3600);
  const m  = Math.floor((sec % 3600) / 60);
  const s  = Math.floor(sec % 60);
  const ms = Math.round((sec - Math.floor(sec)) * 1000);
  const pad = (x, n = 2) => String(x).padStart(n, '0');
  return `${pad(h)}:${pad(m)}:${pad(s)},${pad(ms, 3)}`;
}

_edMergeSelected(){
  const ed = this.editing || {};
  if (!ed.selected || ed.selected.size < 2) {
    return this.showNotification('Pilih minimal 2 baris untuk merge.', 'warning');
  }

  // Ambil rows terpilih
  const idxs = [...ed.selected].sort((a,b)=>a-b);
  const rows = (ed.rows || []);
  const selRows = rows.filter(r => idxs.includes(r.index));
  if (selRows.length < 2) return;

  // Hitung waktu & teks
  selRows.forEach(r => { // pastikan ada detik
    if (r.startS == null || r.endS == null) { 
      r.startS = this._edToSec(r.start); 
      r.endS   = this._edToSec(r.end); 
    }
  });

  const minStartS = Math.min(...selRows.map(r=>r.startS||0));
  const maxEndS   = Math.max(...selRows.map(r=>r.endS||0));

  // Gabung teks; hilangkan newline → 1 baris
  const mergedText = selRows
    .map(r => String(r.translation||'').replace(/\s+/g,' ').trim())
    .filter(Boolean)
    .join(' ');

  // Baris hasil = pakai baris pertama terpilih (index terkecil)
  const keepIdx = idxs[0];
  const keepRow = rows.find(r => r.index === keepIdx);

  // Update baris yang dipertahankan
  keepRow.start  = this._edSecToSrt(minStartS);
  keepRow.end    = this._edSecToSrt(maxEndS);
  keepRow.startS = minStartS;
  keepRow.endS   = maxEndS;
  keepRow.translation = mergedText;

  // Hapus baris lain yang digabung
  const removeSet = new Set(idxs.slice(1));
  ed.rows = rows.filter(r => !removeSet.has(r.index));

  // Bersihkan pilihan & re-render
  ed.selected.clear();
  this._edIndexSeconds();
  this._edFilterRender();
  this._edAttachVttTrack();
  this._edInitBulkSimple?.();
  this.showNotification(`Merged ${idxs.length} rows → #${keepIdx}`, 'success');
}

// Pasang event untuk "Speaker: [input]  [Set]" — aman dipanggil berulang
_edInitBulkSimple() {
  const $ = (id) => document.getElementById(id);
  const btn = $('ed-bulk-set-speaker');
  const inp = $('ed-bulk-speaker');

  // elemen belum ada? skip
  if (!btn || !inp) return;
  if (btn._bound) return;  // cegah double-bind
  btn._bound = true;

  const doSet = () => {
    const name = (inp.value || '').trim();
    if (!name) { this.showNotification?.('Isi nama speaker dulu.', 'warning'); return; }
    this._edBulkSetSpeaker(name);   // langkah 4
  };

  btn.addEventListener('click', doSet);
  inp.addEventListener('keydown', (e) => { if (e.key === 'Enter') doSet(); });
}


/* ================== CC (WebVTT) ================== */

// "00:00:00,880" -> "00:00:00.880"
_srtToVttTime(t) {
  const m = String(t || '').match(/^(\d{2}):(\d{2}):(\d{2}),(\d{1,3})$/);
  if (!m) return '00:00:00.000';
  const ms = String(Number(m[4] || 0)).padStart(3, '0');
  return `${m[1]}:${m[2]}:${m[3]}.${ms}`;
}

// Susun VTT dgn blok STYLE dan cue settings pakai persentase
_buildVTTFromRows(rows, opts = {}) {
  const o = Object.assign({}, this._vttOpts(), opts);

  const out = [
    'WEBVTT',
    '',
    'STYLE',
    `::cue {
      color: ${o.color} !important;
      background-color: ${o.bg} !important;
      font-size: ${o.fontSize} !important;
      font-weight: ${o.weight} !important;
      text-shadow: ${o.shadow};
      line-height: ${o.lineHeight};
      padding: .15em .35em;
      border-radius: .25em;
    }`,
    ''
  ];

  (rows || []).forEach(r => {
    const start = this._srtToVttTime(r.start);
    const end   = this._srtToVttTime(r.end);
    const spk   = (r.speaker || '').trim();
	const gRaw = (r.gender || '').trim().toLowerCase();
    let text    = (r.translation || '').replace(/\r?\n/g, ' ').trim() || '-';
    if (o.withSpeaker && gRaw) text = `${gRaw}: \n${text}`;

    // snapLines=true → line:<integer> (fleksibel); false → line:<percent>
    const lineToken = o.snapLines ? `line:${o.line}` : `line:${o.linePct}%`;
    out.push(`${start} --> ${end} ${lineToken} position:${o.positionPct}% size:${o.sizePct}% align:${o.align}`);
    out.push(text);
    out.push('');
  });

  return out.join('\n');
}

_vttOpts() {
  if (!this.vttOpts) this.vttOpts = {
    // POSISI (fleksibel → multi-line)
    snapLines: true,   // true => pakai "line:<integer>"
    line: -4,          // -4 artinya 4 baris dari bawah (negatif = hitung dari bawah)
    linePct: 86,       // fallback jika snapLines=false (tidak dipakai saat true)
    positionPct: 50,
    sizePct: 88,
    align: 'center',
    withSpeaker: true,

    // STYLING
    color: '#FFD700',                 // emas
    bg: 'rgba(0,0,0,.70)',            // background cue
    fontSize: 'clamp(14px,2.2vw,28px)',
    weight: 400,
    shadow: '0 2px 3px rgba(0,0,0,.9)',
    lineHeight: 1.25,
  };
  return this.vttOpts;
}


_edAttachVttTrack() {
  const v = document.getElementById('ed-video');
  const rows = (this.editing?.rows) || [];
  if (!v || !rows.length) return;

  // bersihkan track sebelumnya
  v.querySelectorAll('track[data-gen="vtt"]').forEach(tr => tr.remove());
  if (this.editing?._vttUrl) {
    try { URL.revokeObjectURL(this.editing._vttUrl); } catch {}
    this.editing._vttUrl = null;
  }

  const vtt = this._buildVTTFromRows(rows, { withSpeaker: true });
  const url = URL.createObjectURL(new Blob([vtt], { type: 'text/vtt' }));

  const tr = document.createElement('track');
  tr.kind = 'subtitles';
  tr.label = 'CC';
  tr.srclang = 'id';
  tr.default = true;
  tr.src = url;
  tr.dataset.gen = 'vtt';
  tr.addEventListener('load', () => { try { tr.track.mode = 'showing'; } catch {} });
  v.appendChild(tr);

  try { v.load(); } catch {}
  this.editing._vttUrl = url;
}

// Debounce kecil kalau nanti kamu panggil rebuild sering
_edRebuildVttSoon() {
  clearTimeout(this._vttTimer);
  this._vttTimer = setTimeout(() => this._edAttachVttTrack(), 250);
}
_edInitVTTDefaults() {
  const o = this._vttOpts();
  // contoh set awal (boleh diubah)
  o.snapLines = true;   // fleksibel multi-line
  o.line = -5;          // naik/turun: -2 lebih bawah, -6 lebih atas
  o.color = '#FFD700';  // warna teks
  o.bg = 'rgba(0,0,0,.70)'; // latar
  this._edAttachVttTrack();
}

_edBindEvents() {
  const $ = id => document.getElementById(id);
  const ed = this.editing;
document.getElementById('ed-warn-only')?.addEventListener('change', () => {
  this.editing.warnOnly = document.getElementById('ed-warn-only').checked;
  this._edFilterRender();
});
  // Auto Sync Timing (geser subtitle vs voice, server-side)
  const autoBtn = document.getElementById('ed-auto-sync');
  if (autoBtn && !autoBtn._bound) {
    autoBtn._bound = true;
    autoBtn.addEventListener('click', async () => {
      await this.edAutoSync();
    });
  }

  $('ed-exp-full')?.addEventListener('click', () => this._edExport('full'));
  $('ed-merge')?.addEventListener('click', () => this._edMergeSelected());

  $('ed-load-session')?.addEventListener('click', async () => {
    const id = $('ed-session-input')?.value.trim() || $('ed-session-select')?.value.trim();
    if (!id) return this.showNotification('Isi/pilih Session ID dulu.', 'warning');
    await this._edLoad(id);
  });

  // APPLY MAPPING → re-fetch dengan policy/majority/snap terbaru
  document.getElementById('ed-apply-mapping')?.addEventListener('click', async () => {
    const sid = this.editing?.sessionId || this.currentSessionId
            || document.getElementById('ed-session-input')?.value?.trim();
    if (!sid) return this.showNotification('Load session dulu.', 'warning');
    await this._edLoad(sid);
  });

  $('ed-session-select')?.addEventListener('change', async (e) => {
    const id = e.target.value;
    if (!id) return;
    $('ed-session-input').value = id;
    await this._edLoad(id);
  });

  $('ed-gender')?.addEventListener('change', () => {
    ed.genderFilter = $('ed-gender').value;
    this._edFilterRender();
  });

  const spkSel = document.getElementById('ed-speaker-select') ||
                 document.getElementById('ed-speaker');
  spkSel?.addEventListener('change', () => {
    this.editing.speakerFilter = spkSel.value || 'all';
    this._edFilterRender();
  });

  $('ed-search')?.addEventListener('input', () => {
    ed.search = $('ed-search').value.trim();
    this._edFilterRender();
  });

  $('ed-follow')?.addEventListener('change', () => { ed.follow = $('ed-follow').checked; });

  $('ed-save')?.addEventListener('click', () => this._edSave());
  $('ed-exp-male')?.addEventListener('click', () => this._edExport('male'));
  $('ed-exp-female')?.addEventListener('click', () => this._edExport('female'));
  $('ed-exp-unk')?.addEventListener('click', () => this._edExport('unknown'));
  $('ed-exp-all')?.addEventListener('click', () => this._edExport('all'));

  $('ed-exp-speaker')?.addEventListener('click', () => {
    const sel = document.getElementById('ed-speaker-select');
    const spk = (sel?.value || '').trim();
    if (!spk || spk === 'all') {
      return this.showNotification('Pilih 1 speaker dulu di dropdown.', 'warning');
    }
    this._edExport('speaker', { speaker: spk });
  });

  $('ed-exp-speaker-zip')?.addEventListener('click', () => {
    this._edExport('speaker_zip');
  });

  const v = $('ed-video');
  v?.addEventListener('timeupdate', () => this._edOnVideoTime(v.currentTime));
  
	const resetBtn = document.getElementById('ed-reset');
	if (resetBtn) {
		resetBtn.addEventListener('click', async () => {
			if (!this.currentSessionId) {
				this.showNotification('No active session', 'error');
				return;
			}

			const ok = confirm(
				'Reset cache? Semua edit manual di tab Editing akan hilang dan akan diganti hasil diarization/translate terbaru.'
			);
			if (!ok) return;

			try {
				this.showLoading('Resetting editing cache...');
				// panggil endpoint backend baru
				const res = await fetch(`/api/session/${this.currentSessionId}/editing/reset`, {
					method: 'POST'
				});

				const txt = await res.text();
				if (!res.ok) {
					this.showNotification(`Reset gagal: ${txt}`, 'error');
					return;
				}

				// setelah cache dihapus, panggil loader bawaan yang biasa dipakai
				// untuk ngisi tabel editing dari server:
				await this._edLoad(this.currentSessionId);

				this.showNotification('Editing cache refreshed from latest diarization / translation', 'success');
			} catch (err) {
				this.showNotification(`Reset error: ${err?.message || err}`, 'error');
			} finally {
				this.hideLoading();
			}
		});
	}

}

async edAutoSync() {
  if (!this.currentSessionId) return this.showNotification('No active session', 'error');
  this.showLoading('Auto-syncing timing to voice…');
  try {
    const res = await fetch(`/api/session/${this.currentSessionId}/editing/auto-sync`, { method: 'POST' });
    const data = await res.json();
    if (!res.ok) throw new Error(data?.detail || 'Failed');
    await this._edLoad(this.currentSessionId); // reload tabel dari server
    this.showNotification(`Auto-sync OK (anchors=${data.anchors}, a=${data.fit.a.toFixed(4)}, b=${data.fit.b_ms}ms)`, 'success');
  } catch (e) {
    this.showNotification(`Auto-sync error: ${e.message||e}`, 'error');
  } finally {
    this.hideLoading();
  }
}



_edPopulateSpeakerFilter(list) {
  const sel =
    document.getElementById('ed-speaker-select') ||
    document.getElementById('ed-speaker');
  if (!sel) return;

  // daftar speaker dari backend (hanya yang non-empty)
  const speakers = Array.isArray(list) ? list.filter(Boolean) : [];

  // hitung baris yang kosong
  const rows = (this.editing?.rows || []);
  const emptyCount = rows.filter(r => !((r.speaker || '').trim())).length;

  const current = (this.editing?.speakerFilter || 'all');

  let html = `<option value="all">All speakers</option>`;
  if (emptyCount > 0) {
    html += `<option value="__EMPTY__">SPEAKER_* (kosong: ${emptyCount})</option>`;
  }
  html += speakers.map(s => `<option value="${this.escapeHtml(s)}">${this.escapeHtml(s)}</option>`).join('');

  sel.innerHTML = html;
  // kalau pilihan sebelumnya masih valid, pertahankan
  const valid = ['all','__EMPTY__',...speakers];
  sel.value = valid.includes(current) ? current : 'all';
}




async _edPopulateSessionSelector() {
  try {
    const res = await fetch('/api/sessions');
    if (!res.ok) return;
    const list = await res.json(); // [{id,...}]
    const sel = document.getElementById('ed-session-select');
    if (!sel) return;
    sel.innerHTML = `<option value="">— pilih session —</option>` +
      list.map(s => `<option value="${s.id}">${s.id}</option>`).join('');
  } catch {}
}

_edHistoryInit() {
  const ed = this.editing || (this.editing = {});
  ed._undo = ed._undo || [];
  ed._redo = ed._redo || [];
  ed._histLimit = 200;

  const btnUndo = document.getElementById('ed-undo');
  const btnRedo = document.getElementById('ed-redo');

  const syncBtns = () => {
    if (btnUndo) btnUndo.disabled = !(ed._undo.length);
    if (btnRedo) btnRedo.disabled = !(ed._redo.length);
  };
  this._edSyncHistoryBtns = syncBtns;
  syncBtns();

  // click
  btnUndo?.addEventListener('click', () => this._edUndo());
  btnRedo?.addEventListener('click', () => this._edRedo());

  // keyboard: Ctrl+Z / Ctrl+Y (atau Ctrl+Shift+Z)
  if (!this._edHistKeyBound) {
    this._edHistKeyBound = true;
    document.addEventListener('keydown', (e) => {
      const k = (e.key || '').toLowerCase();
      if (e.ctrlKey && !e.shiftKey && k === 'z') { e.preventDefault(); this._edUndo(); }
      else if ((e.ctrlKey && k === 'y') || (e.ctrlKey && e.shiftKey && k === 'z')) {
        e.preventDefault(); this._edRedo();
      }
    });
  }
}

async _edLoad(sessionId) {
  const ed = this.editing;
  this.showLoading('Loading editing data…');
  try {
    const p = new URLSearchParams({
      assign_policy: document.getElementById('assign-policy')?.value || 'adaptive',
      majority: String(parseFloat(document.getElementById('majority')?.value || '0.60')),
      snap_ms: String(parseInt(document.getElementById('snap-ms')?.value || '120', 10)),
      // KUNCI: eksplisitkan pisah jalur gender
      gender_mode: 'segment_only',
    });
    const url = `/api/session/${encodeURIComponent(sessionId)}/editing?${p.toString()}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(await res.text());

    const data = await res.json(); // { video, rows:[...], speakers? }
    ed.sessionId = sessionId;
    this.currentSessionId = sessionId;

    ed.rows = (data.rows || []).map(r => ({
      index: r.index,
      start: r.start,
      end: r.end,
      translation: r.translation || "",
      speaker: r.speaker || "",
      gender: (r.gender || "unknown").toLowerCase(),
      notes: r.notes || "",
	  // --- warning fields ---
	  warn_level: r.warn_level || "OK",
	  warn_codes: Array.isArray(r.warn_codes) ? r.warn_codes : [],
	  frac_spk: r.frac_spk ?? null,
	  frac_gen: r.frac_gen ?? null,
	  near_start_ms: r.near_start_ms ?? null,
	  near_end_ms: r.near_end_ms ?? null,
	  dur_ms: r.dur_ms ?? null
	}));

    this._edIndexSeconds();

    // Speakers dari backend; kalau kosong, derive dari rows
    ed.speakers = Array.isArray(data.speakers) && data.speakers.length
      ? data.speakers
      : [...new Set(ed.rows.map(r => (r.speaker || '').trim()).filter(Boolean))].sort();

    this._edPopulateSpeakerFilter(ed.speakers);
    ed.videoUrl = `/api/session/${encodeURIComponent(sessionId)}/video`;

    const info = document.getElementById('ed-session-info');
    if (info) info.textContent = `Session: ${sessionId}`;
    const video = document.getElementById('ed-video');
    if (video) { video.src = ed.videoUrl; }
    this._edSyncListHeightToVideo();
    this._edInitVTTDefaults();
    this._edAttachVttTrack();

    // seed filter awal
    if (!this.editing.genderFilter)  this.editing.genderFilter  = 'all';
    if (!this.editing.speakerFilter) this.editing.speakerFilter = 'all';
    if (typeof this.editing.search !== 'string') this.editing.search = '';

    // render pertama
    this._edFilterRender();
    this._edInitBulkUI();
    this._edLoadSpeakerColors?.();
    this._edHistoryInit?.();
	if (typeof this.editing.warnOnly !== 'boolean') this.editing.warnOnly = false;
    this.showNotification('Editing data loaded.', 'success');
  } catch (e) {
    this.showNotification(`Load editing gagal: ${e.message || e}`, 'error');
  } finally {
    this.hideLoading();
  }
}


// Patch item: { idx, key: 'gender'|'speaker'|'translation', before, after }

_edPushHistory(label, patches) {
  const ed = this.editing || (this.editing = {});
  if (!patches || !patches.length) return;
  ed._undo.push({ label, patches });
  if (ed._undo.length > (ed._histLimit||200)) ed._undo.shift();
  ed._redo.length = 0; // clear redo on new op
  this._edSyncHistoryBtns?.();
}

_edApplyPatches(patches, dir /*'undo'|'redo'*/) {
  const ed = this.editing || {};
  let needRerender = false;

  for (const p of patches) {
    const row = (ed.rows || []).find(x => x.index === p.idx);
    if (!row) { needRerender = true; continue; }
    const val = (dir === 'undo') ? p.before : p.after;
    row[p.key] = val;

    // update DOM kalau ada
    const el = document.getElementById(`row-${p.idx}`);
    if (!el) { needRerender = true; continue; }

    if (p.key === 'translation') {
      const inp = el.querySelector('.ed-text');
      if (inp && inp.value !== val) inp.value = val || '';
    } else if (p.key === 'speaker') {
      const inp = el.querySelector('.ed-speaker');
      if (inp && inp.value !== (val||'')) {
        inp.value = val || '';
        // jika kamu pakai pewarnaan speaker:
        if (this._edEnsureSpeakerColor) {
          const { solid, bg } = this._edEnsureSpeakerColor(val);
          inp.style.backgroundColor = bg;
          inp.style.borderColor = solid;
        }
      }
    } else if (p.key === 'gender') {
      // support SELECT native:
      const sel = el.querySelector('.ed-gender');
      if (sel) {
        sel.value = val || 'unknown';
        // update strip & bg
        const gSolid = this._genderBorder?.(val);
        const gBg    = this._genderBg?.(val);
        if (gBg)  sel.style.backgroundColor = gBg;
        if (gSolid) {
          sel.style.borderColor = gSolid;
          el.style.boxShadow = `inset 4px 0 0 ${gSolid}`;
        }
      }
      // support custom popover button:
      const gbtn = el.querySelector('.ed-gbtn .ed-glabel') ? el.querySelector('.ed-gbtn') : null;
      if (gbtn) {
        const glabel = el.querySelector('.ed-glabel');
        if (glabel) glabel.textContent = val || 'unknown';
        const gSolid = this._genderBorder?.(val);
        const gBg    = this._genderBg?.(val);
        if (gBg)  gbtn.style.backgroundColor = gBg;
        if (gSolid) {
          gbtn.style.borderColor = gSolid;
          el.style.boxShadow = `inset 4px 0 0 ${gSolid}`;
        }
      }
    }
  }

  if (needRerender) this._edRenderList?.(); // fallback jika DOM baris tidak ada
}

_edUndo() {
  const ed = this.editing || {};
  const rec = ed._undo?.pop();
  if (!rec) return;
  // terapkan BEFORE
  this._edApplyPatches(rec.patches, 'undo');
  // simpan ke redo
  ed._redo.push(rec);
  this._edSyncHistoryBtns?.();
}

_edRedo() {
  const ed = this.editing || {};
  const rec = ed._redo?.pop();
  if (!rec) return;
  // terapkan AFTER
  this._edApplyPatches(rec.patches, 'redo');
  // kembali ke undo
  ed._undo.push(rec);
  this._edSyncHistoryBtns?.();
}

_edWarnBadge(row) {
  const level = row?.warn_level || 'OK';
  if (level === 'OK') return '';
  const color = level === 'ALERT' ? 'bg-red-600' : 'bg-amber-500';
  const text  = level === 'ALERT' ? 'ALERT'      : 'WARN';
  const tip   = Array.isArray(row?.warn_codes) && row.warn_codes.length
    ? row.warn_codes.join(', ')
    : text;
  return `<span title="${tip}" class="ml-1 inline-block text-[10px] ${color} text-white px-2 py-0.5 rounded">${text}</span>`;
}

_edWarnStyleAndTitle(row) {
  const level = row?.warn_level || 'OK';
  if (level === 'OK') return { style: '', title: '' };

  const color = (level === 'ALERT') ? '#dc2626' /* red-600 */ : '#f59e0b' /* amber-500 */;
  const st = `border-color:${color}; box-shadow: inset 0 0 0 1px ${color};`;
  const spk = (row?.frac_spk != null) ? `spk ${(row.frac_spk*100).toFixed(0)}%` : '';
  const gen = (row?.frac_gen != null) ? `gen ${(row.frac_gen*100).toFixed(0)}%` : '';
  const ns  = (row?.near_start_ms != null) ? `Δstart ${row.near_start_ms}ms` : '';
  const ne  = (row?.near_end_ms   != null) ? `Δend ${row.near_end_ms}ms`     : '';
  const codes = Array.isArray(row?.warn_codes) ? row.warn_codes.join(', ') : '';
  const title = [level, codes, spk, gen, ns, ne].filter(Boolean).join(' | ');
  return { style: st, title };
}


_edFilterRender(){
  const ed = this.editing || {};
  const rows = ed.rows || [];
  const term = (ed.search || '').toLowerCase();

  const gf = (ed.genderFilter || 'all'); // 'all' | 'male' | 'female' | 'unknown'
  const sf = (ed.speakerFilter || 'all'); // 'all' | '__EMPTY__' | 'SPEAKER_*'

  // filter dasar → ke ed.filtered
  ed.filtered = rows.filter(r => {
    const okG = (gf === 'all') || ((r.gender || 'unknown') === gf);
    const okS = (sf === 'all')
      || (sf === '__EMPTY__' ? !((r.speaker || '').trim())
                             : ((r.speaker || '') === sf));
    const okQ = !term || (r.translation || '').toLowerCase().includes(term)
                      || (r.text || '').toLowerCase().includes(term);
    return okG && okS && okQ;
  });

  // ⟵ APPLY WARNINGS ONLY KE ed.filtered (BUKAN ke rows)
  if (ed.warnOnly) {
    ed.filtered = ed.filtered.filter(r => (r.warn_level || 'OK') !== 'OK');
  }

  const counter = document.getElementById('ed-counter');
  if (counter) counter.textContent = `${ed.filtered.length} shown / ${rows.length} total`;

  this._edRenderList();
  this._edInitBulkSimple?.();
}



_edRenderList() {
  const wrap = document.getElementById('ed-list');
  const ed = this.editing;
  if (!wrap) return;
  wrap.style.overflowY = 'auto';

  // atur tinggi maksimum otomatis (menggunakan tinggi viewport)
  const rect = wrap.getBoundingClientRect();
  const available = window.innerHeight - rect.top - 24; // 24px margin bawah
  if (available > 200) wrap.style.maxHeight = available + 'px';

  // update saat window di-resize
  if (!this._edListResizeBound) {
    this._edListResizeBound = () => {
      const r = wrap.getBoundingClientRect();
      const avail = window.innerHeight - r.top - 24;
      if (avail > 200) wrap.style.maxHeight = avail + 'px';
    };
    window.addEventListener('resize', this._edListResizeBound);
  }
//style="grid-template-columns:28px 48px 220px 90px speaker=85px transl=1fr play =38px; box-shadow: inset 4px 0 0 ${gSolid};"

  if (!ed.filtered.length) { wrap.innerHTML = `<div class="p-4 text-gray-400">No rows.</div>`; return; }
  const esc = s => String(s||'').replace(/[&<>"']/g,c=>({ '&':'&','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[c]));

	// 1 baris: [#id][time][gender][speaker][translation][play]
	wrap.innerHTML = ed.filtered.map(r => {
	  const gSolid = this._genderBorder(r.gender);
	  const gBg    = this._genderBg(r.gender);

	  // menu 3 opsi gender
	  const gMenu = ['male','female','unknown'].map(g => {
		const s = this._genderBorder(g);
		const b = this._genderBg(g);
		return `<button class="ed-gpick block w-full text-left rounded px-2 py-1 text-xs mb-1"
						data-idx="${r.index}" data-g="${g}"
						style="background-color:${b}; color:#E5E7EB; border:1px solid ${s}">
				  ${g}
				</button>`;
	  }).join('');

	  // WARN: style & title untuk kolom time
	  const { style: warnTimeStyle, title: warnTimeTitle } = this._edWarnStyleAndTitle(r);

	  return `
		<div class="ed-row grid items-center gap-2"
			 style="grid-template-columns:28px 38px 90px 85px 1fr 48px 220px; box-shadow: inset 4px 0 0 ${gSolid};"
			 data-idx="${r.index}" id="row-${r.index}">
		  <input type="checkbox" class="ed-chk" data-idx="${r.index}" ${ this.editing?.selected?.has(r.index) ? 'checked' : '' }>

		  <button class="ed-play btn btn-slate px-2 py-1 text-xs">▶</button>

		  <!-- GENDER: custom dropdown -->
		  <div class="ed-gwrap relative">
			<button class="ed-gbtn border rounded text-xs px-2 py-1 w-[80px] flex items-center justify-between"
					data-idx="${r.index}"
					style="background-color:${gBg}; color:#E5E7EB; border-color:${gSolid}">
			  <span class="ed-glabel">${r.gender||'unknown'}</span>
			  <span class="ml-1 opacity-70">▾</span>
			</button>
			<div class="ed-gmenu hidden absolute z-20 right-0 mt-1 p-1 rounded border border-gray-600 bg-gray-800 shadow-lg w-[120px]">
			  ${gMenu}
			</div>
		  </div>

		  <!-- SPEAKER tetap -->
		  <input class="ed-speaker border rounded text-xs w-full px-2 py-1"
				 style="background-color:${this._edEnsureSpeakerColor(r.speaker).bg}; color:#F9FAFB; border-color:${this._edEnsureSpeakerColor(r.speaker).solid}"
				 value="${esc(r.speaker||'')}" placeholder="SPEAKER_*"/>

		  <!-- TRANSLATION -->
		  <input class="ed-text bg-gray-800 border border-gray-700 text-green-400 rounded px-2 py-1 text-sm w-full
						whitespace-nowrap overflow-hidden text-ellipsis"
				 value="${esc(r.translation||'')}" placeholder="Translation…"/>

		  <!-- #ID + WARN badge -->
		  <div class="text-gray-400 font-mono flex items-center">
			#${r.index} ${this._edWarnBadge(r)}   <!-- WARN -->
		  </div>
		  
		  <!-- TIME: diberi outline + tooltip saat WARN -->
		  <div class="ed-time font-mono text-sm bg-gray-800 border border-gray-700 rounded px-2 py-1 whitespace-nowrap"
			   style="${warnTimeStyle}" title="${warnTimeTitle}">
			${r.start} <span class="opacity-60">→</span> ${r.end}
		  </div>
		</div>
	  `;
	}).join('');


// toggle buka/tutup menu
wrap.querySelectorAll('.ed-gbtn').forEach(btn => {
  btn.addEventListener('click', (e) => {
    e.stopPropagation();
    const gwrap = e.currentTarget.closest('.ed-gwrap');
    const menu  = gwrap.querySelector('.ed-gmenu');
    // tutup menu lain
    wrap.querySelectorAll('.ed-gmenu').forEach(m => { if (m!==menu) m.classList.add('hidden'); });
    menu.classList.toggle('hidden');
  });
});

// pilih gender
wrap.querySelectorAll('.ed-gpick').forEach(item => {
  item.addEventListener('click', (e) => {
    e.stopPropagation();
    const idx = Number(e.currentTarget.dataset.idx);
    const val = e.currentTarget.dataset.g;
    const row = this.editing.rows.find(x => x.index === idx);
    if (!row) return;

    const before = row.gender || 'unknown';   // <-- HISTORY (before)
    row.gender = val;

    const gwrap = e.currentTarget.closest('.ed-gwrap');
    const gbtn  = gwrap.querySelector('.ed-gbtn');
    const glabel= gwrap.querySelector('.ed-glabel');
    const gSolid= this._genderBorder(val);
    const gBg   = this._genderBg(val);
    glabel.textContent = val;
    gbtn.style.backgroundColor = gBg;
    gbtn.style.borderColor     = gSolid;
    gwrap.querySelector('.ed-gmenu').classList.add('hidden');

    const rowEl = e.currentTarget.closest('.ed-row');
    rowEl.style.boxShadow = `inset 4px 0 0 ${gSolid}`;

    if (before !== val) {                    // <-- HISTORY (push)
      this._edPushHistory(`Set gender (#${idx})`, [{ idx, key:'gender', before, after: val }]);
    }
  });
});


// klik di luar → tutup semua menu
if (!this._edGenderOutsideBound) {
  this._edGenderOutsideBound = (ev) => {
    if (!wrap.contains(ev.target)) {
      wrap.querySelectorAll('.ed-gmenu').forEach(m => m.classList.add('hidden'));
    }
  };
  document.addEventListener('click', this._edGenderOutsideBound);
}

	this._edSyncListHeightToVideo();
	ed.selected = ed.selected || new Set();
	wrap.querySelectorAll('.ed-chk').forEach(chk => {
	  chk.addEventListener('change', () => {
		const idx = Number(chk.dataset.idx);
		if (chk.checked) ed.selected.add(idx); else ed.selected.delete(idx);
		this._edUpdateMergeBtn();
	  });
	});
      this._edUpdateSelCount();
      this._edUpdateMasterChk();
      this._edUpdateMergeBtn();

	wrap.querySelectorAll('.ed-gender').forEach(sel=>{
	  sel.addEventListener('change', e=>{
		const idx = Number(e.target.closest('[data-idx]').dataset.idx);
		const row = this.editing.rows.find(x=>x.index===idx);
		if (!row) return;
		row.gender = e.target.value;

		const gSolid = this._genderBorder(row.gender);
		const gBg    = this._genderBg(row.gender);
		sel.style.backgroundColor = gBg;         // penting: backgroundColor
		sel.style.borderColor     = gSolid;
		sel.style.color           = '#E5E7EB';
	  });
	});

// TRANSLATION
wrap.querySelectorAll('.ed-text').forEach(inp=>{
  // simpan nilai awal saat fokus
  inp.addEventListener('focus', e=>{
    const idx = Number(e.target.closest('[data-idx]').dataset.idx);
    e.target.dataset._orig = this.editing.rows.find(x=>x.index===idx)?.translation || '';
  });
  // update model realtime
  inp.addEventListener('input', e=>{
    const idx = Number(e.target.closest('[data-idx]').dataset.idx);
    const row = this.editing.rows.find(x=>x.index===idx);
    if (row) row.translation = e.target.value;
  });
  // saat blur → push ke history kalau berubah
  inp.addEventListener('blur', e=>{
    const idx = Number(e.target.closest('[data-idx]').dataset.idx);
    const row = this.editing.rows.find(x=>x.index===idx);
    if (!row) return;
    const before = e.target.dataset._orig ?? '';
    const after  = e.target.value || '';
    if (before !== after) {
      this._edPushHistory(`Edit text (#${idx})`, [{ idx, key:'translation', before, after }]);
    }
  });
});

// SPEAKER
wrap.querySelectorAll('.ed-speaker').forEach(inp=>{
  inp.addEventListener('focus', e=>{
    const idx = Number(e.target.closest('[data-idx]').dataset.idx);
    e.target.dataset._orig = this.editing.rows.find(x=>x.index===idx)?.speaker || '';
  });
  inp.addEventListener('input', e=>{
    const idx = Number(e.target.closest('[data-idx]').dataset.idx);
    const row = this.editing.rows.find(x=>x.index===idx);
    if (!row) return;
    row.speaker = e.target.value;

    // pewarnaan langsung
    const { solid, bg } = this._edEnsureSpeakerColor(row.speaker);
    e.target.style.backgroundColor = bg;
    e.target.style.borderColor     = solid;
  });
  inp.addEventListener('blur', e=>{
    const idx = Number(e.target.closest('[data-idx]').dataset.idx);
    const row = this.editing.rows.find(x=>x.index===idx);
    if (!row) return;
    const before = e.target.dataset._orig ?? '';
    const after  = e.target.value || '';
    if (before !== after) {
      this._edPushHistory(`Set speaker (#${idx})`, [{ idx, key:'speaker', before, after }]);
    }
  });
});


  wrap.querySelectorAll('.ed-play').forEach(btn=>{
    btn.addEventListener('click', e=>{
      const idx = Number(e.target.closest('[data-idx]').dataset.idx);
      this._edSeekToRow(idx, true);
    });
  });
  this._edUpdateMasterChk?.();
  this._edInitBulkSimple?.();
}

// app.js — final version
_edSeekToRow(idx, play=false){
  const ed=this.editing; const v=document.getElementById('ed-video');
  const r = ed.rows.find(x=>x.index===idx); if(!r||!v) return;
  v.currentTime = r.startS + 0.001;
  this._edHighlight(idx, true);
  if (play){ ed.playUntil = r.endS; v.play(); }
}

_edTargetIndicesSelected() {
  const ed = this.editing || {};
  return Array.from(ed.selected || new Set());
}

_edRebuildSpeakerFilterOptions() {
  const sel = document.getElementById('ed-speaker');
  if (!sel) return;
  const speakers = new Set(
    (this.editing.rows || [])
      .map(r => (r.speaker || '').trim())
      .filter(Boolean)
  );
  sel.innerHTML = `<option value="">All speakers</option>` +
    [...speakers].sort().map(s => `<option value="${this.escapeHtml(s)}">${this.escapeHtml(s)}</option>`).join('');

  // when user changes selection, filter like gender
  sel.onchange = () => { 
    this.editing.speakerFilter = sel.value.trim().toLowerCase(); 
    this._edFilterRender(); 
  };
}

// Samakan tinggi panel list (#ed-list) dengan tinggi video (#ed-video)
_edSyncListHeightToVideo() {
  const list  = document.getElementById('ed-list');
  const video = document.getElementById('ed-video');
  if (!list || !video) return;

  const apply = () => {
    const h = Math.round(video.getBoundingClientRect().height);
    if (h > 0) {
      list.style.height    = h + 'px';
      list.style.maxHeight = h + 'px';
      list.style.overflowY = 'auto';
    }
  };

  // apply sekarang
  apply();

  // update saat metadata/bitmap masuk
  if (!video._ccMetaBound) {
    video.addEventListener('loadedmetadata', apply);
    video.addEventListener('loadeddata', apply);
    video._ccMetaBound = true;
  }

  // update kalau ukuran video berubah (responsive/layout berubah)
  if (!this._videoResizeObs) {
    try {
      this._videoResizeObs = new ResizeObserver(apply);
      this._videoResizeObs.observe(video);
    } catch {
      // fallback kalau ResizeObserver tidak ada
      window.addEventListener('resize', apply);
    }
  }
}

_edOnVideoTime(t){
  const ed=this.editing; if(!ed?.rows?.length) return;
  const r = ed.rows.find(x=>t>=x.startS && t<x.endS);
  if (r) this._edHighlight(r.index, true);

  // stop di akhir segmen jika sedang "play segment"
  if (ed.playUntil != null && t>=ed.playUntil-0.02) {
    const v=document.getElementById('ed-video'); v?.pause();
    ed.playUntil = null;
  }
}

_edToSec(t){ const m=t?.match?.(/(\d+):(\d+):(\d+),(\d+)/); if(!m) return 0;
  return (+m[1])*3600+(+m[2])*60+(+m[3])+((+m[4])/1000); }

// panggil sekali saat selesai _edLoad(): tambahkan startS/endS ke setiap row
_edIndexSeconds(){
  (this.editing.rows || []).forEach(r=>{
    const m = (t)=> {
      const x = (t||'').match(/(\d+):(\d+):(\d+),(\d+)/);
      return x ? (+x[1])*3600+(+x[2])*60+(+x[3])+(+x[4])/1000 : 0;
    };
    r.startS = r.startS ?? m(r.start);
    r.endS   = r.endS   ?? m(r.end);
  });
}

// GANTI seluruh fungsi _edHighlight dengan versi ini
_edHighlight(idx, doFollow = true) {
  const ed = this.editing; if (!ed) return;
  if (ed.liveIndex === idx) return;

  // hapus highlight lama & set yang baru
  if (ed.liveIndex) document.getElementById(`row-${ed.liveIndex}`)?.classList.remove('live');
  ed.liveIndex = idx;
  const rowEl = document.getElementById(`row-${idx}`);
  rowEl?.classList.add('live');

  // HANYA scroll kontainer list, bukan halaman
  if (doFollow && ed.follow && rowEl) {
    const list = document.getElementById('ed-list');
    if (list) {
      // posisi row relatif ke kontainer
      const rowTopInList = rowEl.offsetTop - list.offsetTop;
      // pusatkan baris di dalam viewport kontainer
      const target = rowTopInList - Math.max(0, (list.clientHeight - rowEl.clientHeight) / 2);
      list.scrollTo({ top: Math.max(0, target), behavior: 'smooth' });
    } else {
      // fallback kalau #ed-list tak ditemukan
      rowEl.scrollIntoView({ block: 'center', behavior: 'smooth' });
    }
  }
}


_edAutoResizeTextareas(){
  document.querySelectorAll('.ed-translation').forEach(t=>{
    const fit = () => { t.style.height='auto'; t.style.height = (t.scrollHeight)+'px'; };
    fit();
    t.addEventListener('input', fit);
  });
}

async _edSave() {
  const ed = this.editing;
  if (!ed.sessionId) return this.showNotification('No session.', 'warning');
  try {
    this.showLoading('Saving editing…');
    const res = await fetch(`/api/session/${ed.sessionId}/editing`, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ rows: ed.rows })
    });
    if (!res.ok) throw new Error(await res.text());
    this.showNotification('Saved.', 'success');
  } catch (e) {
    this.showNotification(`Save failed: ${e.message || e}`, 'error');
  } finally {
    this.hideLoading();
  }
}

async _edExport(mode, extra = {}) {
  const ed = this.editing;
  if (!ed.sessionId) return this.showNotification('No session.', 'warning');
  try {
    this.showLoading('Exporting…');
    const res = await fetch(`/api/session/${ed.sessionId}/editing/export`, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ mode, reindex: true, ...extra })
    });
    if (!res.ok) throw new Error(await res.text());

    const disp = res.headers.get('Content-Disposition') || '';
    let fname = 'export.srt';
    const m = disp.match(/filename="?([^"]+)"?/i);
    if (m) fname = m[1];

    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = fname;
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);

    this.showNotification('Export done.', 'success');
  } catch (e) {
    this.showNotification(`Export failed: ${e.message || e}`, 'error');
  } finally {
    this.hideLoading();
  }
}



    // ============================
    // TAB 4: TTS EXPORT
    // ============================

    async loadTTSExportTab() {
        console.log('Loading TTS Export tab...');
        const tabContent = document.getElementById('tts-export');
        if (!tabContent) return;

        // Clear and show loading
        tabContent.innerHTML = `
            <div class="bg-gray-800 rounded-lg p-6">
                <h2 class="text-2xl font-bold mb-6 text-purple-400">
                    <i class="fas fa-file-export mr-2"></i>TTS & Export
                </h2>
                <div class="text-center py-8">
                    <div class="loading-spinner mx-auto mb-4"></div>
                    <p class="text-gray-400">Loading TTS & Export interface...</p>
                </div>
            </div>
        `;

        setTimeout(() => {
            this.renderTTSExportTab();
        }, 500);
    }

    renderTTSExportTab() {
        const tabContent = document.getElementById('tts-export');
        if (!tabContent) return;

        tabContent.innerHTML = `
            <div class="bg-gray-800 rounded-lg p-6">
                <h2 class="text-2xl font-bold mb-6 text-purple-400">
                    <i class="fas fa-file-export mr-2"></i>TTS & Export
                </h2>

                <div class="bg-gray-700 rounded-lg p-6">
                    <div class="text-center mb-6">
                        <i class="fas fa-robot text-6xl text-purple-400 mb-4"></i>
                        <h3 class="text-xl font-semibold mb-2 text-white">TTS & Export Interface</h3>
                        <p class="text-gray-300 mb-6">Text-to-speech and video export functionality</p>
                    </div>
                    
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                        <div class="text-center p-4 bg-gray-600 rounded-lg">
                            <i class="fas fa-edge text-4xl text-blue-400 mb-3"></i>
                            <div class="text-lg font-semibold text-white mb-2">Edge TTS</div>
                            <div class="text-gray-300">Free • Fast • Good Quality</div>
                            <button class="mt-3 bg-blue-500 hover:bg-blue-600 px-4 py-2 rounded transition-colors">
                                Select Engine
                            </button>
                        </div>
                        <div class="text-center p-4 bg-gray-600 rounded-lg">
                            <i class="fas fa-star text-4xl text-purple-400 mb-3"></i>
                            <div class="text-lg font-semibold text-white mb-2">ElevenLabs</div>
                            <div class="text-gray-300">Premium • Best Quality • Natural</div>
                            <button class="mt-3 bg-purple-500 hover:bg-purple-600 px-4 py-2 rounded transition-colors">
                                Select Engine
                            </button>
                        </div>
                    </div>

                    <div class="text-center">
                        <button class="bg-green-500 hover:bg-green-600 px-6 py-3 rounded-lg text-lg font-semibold transition-colors">
                            <i class="fas fa-play-circle mr-2"></i>Start TTS & Export
                        </button>
                    </div>
                </div>
            </div>
        `;

        this.setupTTSExportEvents();
    }

    setupTTSExportEvents() {
        const buttons = document.querySelectorAll('#tts-export button');
        buttons.forEach(btn => {
            btn.addEventListener('click', () => {
                this.showNotification(`${btn.textContent} functionality coming soon`, 'info');
            });
        });
    }

    // === BACKEND METHODS ===
    async testBackendConnection() {
        try {
            const response = await fetch('/api/test');
            if (response.ok) {
                const data = await response.json();
                console.log('Backend test:', data);
                this.showNotification('Backend connected successfully', 'success');
                this.updateConnectionStatus(true);
            } else {
                throw new Error('Test endpoint failed');
            }
        } catch (error) {
            console.error('Backend connection failed:', error);
            this.showNotification('Cannot connect to backend', 'error');
            this.updateConnectionStatus(false);
        }
    }

    async createTestSession() {
        this.showLoading('Creating test session...');
        
        try {
            const response = await fetch('/api/session/create', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: 'video_name=test_video.mp4&srt_name=test_subtitles.srt'
            });

            if (response.ok) {
                const data = await response.json();
                this.currentSessionId = data.session_id;
                this.showNotification('Test session created successfully', 'success');
                this.updateSessionInfo(data.session_id);
            } else {
                throw new Error('Failed to create session');
            }
        } catch (error) {
            console.error('Error creating session:', error);
            this.showNotification('Error creating session', 'error');
        } finally {
            this.hideLoading();
        }
    }

    async createSession() {
        this.showLoading('Creating session...');
        
        try {
            const response = await fetch('/api/session/create', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: 'video_name=uploaded_video.mp4&srt_name=uploaded_subtitles.srt'
            });

            if (response.ok) {
                const data = await response.json();
                this.currentSessionId = data.session_id;
                this.showNotification('Session created successfully', 'success');
                this.updateSessionInfo(data.session_id);
            } else {
                throw new Error('Failed to create session');
            }
        } catch (error) {
            console.error('Error creating session:', error);
            this.showNotification('Error creating session', 'error');
        } finally {
            this.hideLoading();
        }
    }

    async loadSessionList() {
        const sessionList = document.getElementById('session-list');
        if (!sessionList) return;

        try {
            const response = await fetch('/api/sessions');
            if (response.ok) {
                const data = await response.json();
                const sessions = data.sessions || [];
                
                if (sessions.length === 0) {
                    sessionList.innerHTML = `
                        <div class="text-gray-400 text-center py-4">
                            <i class="fas fa-inbox text-2xl mb-2"></i>
                            <div>No sessions found</div>
                        </div>
                    `;
                    return;
                }

                sessionList.innerHTML = sessions.map(session => `
                    <div class="session-item p-3 rounded cursor-pointer hover:bg-gray-600 border border-gray-600 transition-colors"
                         data-session-id="${session.id}">
                        <div class="font-medium text-sm text-white">${session.video_name || 'Untitled Session'}</div>
                        <div class="text-xs text-gray-400 mt-1">
                            ${new Date(session.created_at * 1000).toLocaleDateString()}
                        </div>
                        <div class="text-xs text-gray-400">
                            Status: <span class="text-${this.getStatusColor(session.status)}">${session.status}</span>
                        </div>
                    </div>
                `).join('');

                // Add click handlers
                sessionList.querySelectorAll('.session-item').forEach(item => {
                    item.addEventListener('click', () => {
                        // Remove active class from all items
                        sessionList.querySelectorAll('.session-item').forEach(i => {
                            i.classList.remove('bg-blue-600', 'border-blue-400');
                        });
                        // Add active class to clicked item
                        item.classList.add('bg-blue-600', 'border-blue-400');
                        
                        this.loadSessionData(item.dataset.sessionId);
                    });
                });

            } else {
                throw new Error('Failed to load sessions');
            }
        } catch (error) {
            sessionList.innerHTML = `
                <div class="text-red-400 text-center py-4">
                    <i class="fas fa-exclamation-triangle mr-2"></i>
                    Error loading sessions
                </div>
            `;
        }
    }

    getStatusColor(status) {
        const colors = {
            'created': 'blue-400',
            'files_uploaded': 'green-400',
            'workdir_ready': 'green-400',
            'audio_ready': 'green-400',
            'diarization_complete': 'green-400',
            'translating': 'yellow-400',
            'translation_complete': 'green-400',
            'tts_export': 'yellow-400',
            'complete': 'green-400',
            'error': 'red-400'
        };
        return colors[status] || 'gray-400';
    }

    async loadSessionData(sessionId) {
        try {
            const response = await fetch(`/api/session/${sessionId}/editing-data`);
            if (response.ok) {
                const data = await response.json();
                this.currentSessionId = sessionId;
                this.updateSessionInfo(sessionId);
                this.showNotification('Session data loaded successfully', 'success');
                
                // You can now render the editing table with data.entries
                console.log('Loaded session data:', data);
            } else {
                throw new Error('Failed to load session data');
            }
        } catch (error) {
            console.error('Error loading session data:', error);
            this.showNotification('Error loading session data', 'error');
        }
    }

    updateSessionInfo(sessionId) {
        const sessionInfo = document.getElementById('current-session-info');
        if (sessionInfo) {
            sessionInfo.innerHTML = `
                <div class="text-sm">
                    <div class="font-medium">Session ID: ${sessionId}</div>
                    <div class="text-gray-400">Created: ${new Date().toLocaleString()}</div>
                </div>
            `;
        }

        const globalSessionInfo = document.getElementById('session-info');
        if (globalSessionInfo) {
            globalSessionInfo.textContent = `Session: ${sessionId.substring(0, 8)}...`;
        }
    }

    updateConnectionStatus(connected) {
        this.isConnected = connected;
        const statusElement = document.getElementById('connection-status');
        if (statusElement) {
            if (connected) {
                statusElement.innerHTML = '<i class="fas fa-circle text-xs mr-1 text-green-400"></i><span class="text-sm">Connected</span>';
            } else {
                statusElement.innerHTML = '<i class="fas fa-circle text-xs mr-1 text-red-400"></i><span class="text-sm">Disconnected</span>';
            }
        }
    }

    showLoading(message = 'Loading...') {
        const loading = document.getElementById('global-loading');
        const messageEl = document.getElementById('loading-message');
        if (loading && messageEl) {
            messageEl.textContent = message;
            loading.classList.remove('hidden');
        }
    }

    hideLoading() {
        const loading = document.getElementById('global-loading');
        if (loading) {
            loading.classList.add('hidden');
        }
    }

    showNotification(message, type = 'info') {
        const container = document.getElementById('notification-container');
        if (!container) return;
        
        const notification = document.createElement('div');
        const bgColor = type === 'error' ? 'bg-red-500' : 
                       type === 'success' ? 'bg-green-500' : 
                       type === 'warning' ? 'bg-yellow-500' : 'bg-blue-500';
        
        notification.className = `${bgColor} text-white p-4 rounded-lg shadow-lg max-w-sm transform transition-transform duration-300 translate-x-full`;
        notification.innerHTML = `
            <div class="flex items-center justify-between">
                <span>${message}</span>
                <button onclick="this.parentElement.parentElement.remove()" class="ml-4">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
        
        container.appendChild(notification);
        
        // Animate in
        setTimeout(() => {
            notification.classList.remove('translate-x-full');
        }, 10);
        
        // Auto remove after 5 seconds
        setTimeout(() => {
            if (notification.parentElement) {
                notification.classList.add('translate-x-full');
                setTimeout(() => {
                    if (notification.parentElement) {
                        notification.remove();
                    }
                }, 300);
            }
        }, 5000);
    }

    saveAll() {
        this.showNotification('All changes saved successfully', 'success');
    }
    
    appendLog(message) {
        const logOutput = document.getElementById('log-output');
        if (logOutput) {
            const logEntry = document.createElement('div');
            logEntry.className = 'text-gray-300 mb-1';
            logEntry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
            logOutput.appendChild(logEntry);
            logOutput.scrollTop = logOutput.scrollHeight;
        }
    }

// ========= OCR EDITING V1 =========
// ========= OK =========
	/* ========= OCR — util kecil ========= */
	_ocrToSec(s){
	  if (typeof this._edToSec === 'function') return this._edToSec(s);
	  const m = String(s||'').match(/(\d+):(\d+):(\d+),(\d+)/);
	  if (!m) return 0;
	  return (+m[1])*3600 + (+m[2])*60 + (+m[3]) + (+m[4])/1000;
	}
	_ocrEsc(s){
	  if (typeof this.escapeHtml === 'function') return this.escapeHtml(String(s||''));
	  return String(s||'').replace(/[&<>\"']/g,c=>({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[c]));
	}

	/* ========= OCR — state ========= */
	initOcrState(){
	  if (this.ocr) return;
	  this.ocr = {
		rows: [], filtered: [],
		search: '', onlyWarn: false, onlyWarnID: false,
		videoUrl: null, playUntil: null, liveIndex: null,
		// CC/OSD controls
		cc: { font: 18, bottomPct: 6, widthPct: 90 }  // px & persen dari bawah
	  };
	}

	/* ========= OCR — UI utama (30:70, tinggi sama, OSD CC + kontrolnya) ========= */
	async loadOcrEditTab(){
	  this.initOcrState();
	  const tab = document.getElementById('ocr-edit'); if (!tab) return;

	  tab.innerHTML = `
	  <div class="bg-gray-800 rounded-lg p-4 lg:p-6">
		<div class="flex items-center justify-between mb-4">
		  <h2 class="text-xl lg:text-2xl font-bold text-blue-400">
			<i class="fas fa-broom mr-2"></i>Editing SRT OCR
		  </h2>
		  <div class="flex items-center gap-2">
		  <!-- CC quick buttons -->
<!-- CC width control -->
<div class="mt-2 grid grid-cols-1 gap-3 text-xs text-gray-300">
    <span class="min-w-[72px]">CC Width</span>
    <input id="cc-width" type="range" min="40" max="100" step="1" class="flex-1" />
    <span id="cc-width-val" class="w-10 text-right"></span>

</div>
			<div class="mt-2 flex items-center gap-2 text-xs text-gray-300">
			  <button id="cc-up"    class="px-2 py-1 rounded bg-gray-800 hover:bg-gray-700">CC ↑</button>
			  <button id="cc-down"  class="px-2 py-1 rounded bg-gray-800 hover:bg-gray-700">CC ↓</button>
			  <button id="cc-plus"  class="px-2 py-1 rounded bg-gray-800 hover:bg-gray-700">+</button>
			  <button id="cc-minus" class="px-2 py-1 rounded bg-gray-800 hover:bg-gray-700">−</button>
			</div>
			<button id="ocr-load-srt" class="px-3 py-2 rounded bg-gray-700 hover:bg-gray-600">
			  <i class="fas fa-file-alt mr-2"></i>Load Subtitle
			</button>
			<button id="ocr-load-video" class="px-3 py-2 rounded bg-gray-700 hover:bg-gray-600">
			  <i class="fas fa-video mr-2"></i>Load Film 9:16
			</button>
			<div class="hidden lg:flex items-center gap-2 ml-2">
			  <input id="ocr-search" type="text" placeholder="Search (text only)"
					 class="px-3 py-2 rounded bg-gray-700 focus:outline-none w-56" />
			  <label class="inline-flex items-center gap-2 select-none text-sm">
				<input id="ocr-only-warn" type="checkbox" class="accent-blue-500" /> warnings
			  </label>
  <label class="inline-flex items-center gap-2 select-none text-sm">
    <input id="ocr-only-warn-id" type="checkbox" class="accent-blue-500" /> warnings ID
  </label>
			</div>
			<button id="ocr-save" class="ml-2 px-3 py-2 rounded bg-green-700 hover:bg-green-600">
			  <i class="fas fa-save mr-2"></i>Save
			</button>
			<button id="ocr-export" class="px-3 py-2 rounded bg-indigo-700 hover:bg-indigo-600">
			  <i class="fas fa-download mr-2"></i>Export
			</button>
		  </div>
		</div>

		<div class="grid grid-cols-1 lg:grid-cols-[3fr_7fr] gap-4">
		  <!-- LEFT: video + CC overlay + kontrol CC -->
		  <div class="bg-gray-700 rounded-lg p-3 h-[calc(100vh-220px)] flex flex-col">
			<div class="relative w-full flex-1 bg-black rounded overflow-hidden flex items-center justify-center">
			  <video id="ocr-video" class="h-full w-auto"
					 style="aspect-ratio:9/16; object-fit:contain;"
					 controls playsinline preload="metadata"></video>
				<div id="ocr-osd"
					 class="absolute left-1/2 -translate-x-1/2 text-center text-white
							bg-black/60 px-3 py-2 rounded pointer-events-none leading-tight"
					 style="word-break: break-word;"></div>
			</div>
			<!-- CC controls -->
			<div class="mt-2 grid grid-cols-2 gap-3 text-xs text-gray-300">
			  <label class="flex items-center gap-2">
				<span class="min-w-[72px]">CC Font</span>
				<input id="cc-font" type="range" min="12" max="40" step="1" class="flex-1" />
				<span id="cc-font-val" class="w-8 text-right"></span>
			  </label>
			  <label class="flex items-center gap-2">
				<span class="min-w-[72px]">CC Pos (↓/↑)</span>
				<input id="cc-pos" type="range" min="0" max="50" step="1" class="flex-1" />
				<span id="cc-pos-val" class="w-8 text-right"></span>
			  </label>
			</div>
		  </div>

		  <!-- RIGHT: list (scroll hanya di kanan) -->
		  <div class="bg-gray-700 rounded-lg p-3 h-[calc(100vh-220px)] flex flex-col">
			<div class="flex lg:hidden items-center gap-2 mb-2">
			  <input id="ocr-search-m" type="text" placeholder="Search (text only)"
					 class="px-3 py-2 rounded bg-gray-800 w-full" />
			  <label class="inline-flex items-center gap-2 text-sm">
				<input id="ocr-only-warn-m" type="checkbox" class="accent-blue-500" /> Only warnings
			  </label>
			    <label class="inline-flex items-center gap-2 text-sm">
    <input id="ocr-only-warn-id-m" type="checkbox" class="accent-blue-500" /> Only warnings ID
  </label>
			</div>
			<div id="ocr-list" class="flex-1 overflow-y-auto"></div>
		  </div>
		</div>
	  </div>`;

	  // set nilai default kontrol CC
	  document.getElementById('cc-font').value = String(this.ocr.cc.font);
	  document.getElementById('cc-font-val').textContent = String(this.ocr.cc.font);
	  document.getElementById('cc-pos').value  = String(this.ocr.cc.bottomPct);
	  document.getElementById('cc-pos-val').textContent  = String(this.ocr.cc.bottomPct)+'%';
	  document.getElementById('cc-width').value = String(this.ocr.cc.widthPct);
	  document.getElementById('cc-width-val').textContent = this.ocr.cc.widthPct + '%';

	  this._ocrBindEvents();
	  this._ocrRefreshOsdStyle();
	  this._ocrRenderList();
	}

	/* ========= OCR — bind events ========= */
	_ocrBindEvents(){
	  const $ = (id) => document.getElementById(id);
	  const v = $('ocr-video');
	  if (v && !v._ocrBound){
		v.addEventListener('timeupdate', ()=> this._ocrOnVideoTime(v.currentTime));
		v._ocrBound = true;
	  }
	const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

	$('cc-up')?.addEventListener('click', ()=>{
	  this.ocr.cc.bottomPct = clamp(this.ocr.cc.bottomPct + 2, 0, 50);  // naik = lebih besar
	  $('cc-pos').value = String(this.ocr.cc.bottomPct);
	  $('cc-pos-val').textContent = String(this.ocr.cc.bottomPct) + '%';
	  this._ocrRefreshOsdStyle();
	});
	$('cc-down')?.addEventListener('click', ()=>{
	  this.ocr.cc.bottomPct = clamp(this.ocr.cc.bottomPct - 2, 0, 50);  // turun = lebih kecil
	  $('cc-pos').value = String(this.ocr.cc.bottomPct);
	  $('cc-pos-val').textContent = String(this.ocr.cc.bottomPct) + '%';
	  this._ocrRefreshOsdStyle();
	});
	$('cc-plus')?.addEventListener('click', ()=>{
	  this.ocr.cc.font = clamp(this.ocr.cc.font + 2, 12, 40);
	  $('cc-font').value = String(this.ocr.cc.font);
	  $('cc-font-val').textContent = String(this.ocr.cc.font);
	  this._ocrRefreshOsdStyle();
	});
	$('cc-minus')?.addEventListener('click', ()=>{
	  this.ocr.cc.font = clamp(this.ocr.cc.font - 2, 12, 40);
	  $('cc-font').value = String(this.ocr.cc.font);
	  $('cc-font-val').textContent = String(this.ocr.cc.font);
	  this._ocrRefreshOsdStyle();
	});
	$('cc-width')?.addEventListener('input', (e)=>{
	  this.ocr.cc.widthPct = Number(e.target.value || 90);
	  $('cc-width-val').textContent = this.ocr.cc.widthPct + '%';
	  this._ocrRefreshOsdStyle();
	});

	  $('ocr-load-srt')?.addEventListener('click', ()=> this._ocrPromptLoadSrt());
	  $('ocr-load-video')?.addEventListener('click', ()=> this._ocrPromptLoadVideo());

	  const onSearch = (e)=>{ this.ocr.search = e.target.value||''; this._ocrRenderList(); };
	  $('ocr-search')?.addEventListener('input', onSearch);
	  $('ocr-search-m')?.addEventListener('input', onSearch);

	  const onOnlyWarn = (e)=>{ this.ocr.onlyWarn = !!e.target.checked; this._ocrRenderList(); };
	  $('ocr-only-warn')?.addEventListener('change', onOnlyWarn);
	  $('ocr-only-warn-m')?.addEventListener('change', onOnlyWarn);
	// (BARU) — Only warnings ID
	const onOnlyWarnID = (e)=>{ this.ocr.onlyWarnID = !!e.target.checked; this._ocrRenderList(); };
	$('ocr-only-warn-id')?.addEventListener('change', onOnlyWarnID);
	$('ocr-only-warn-id-m')?.addEventListener('change', onOnlyWarnID);

	  $('ocr-save')?.addEventListener('click', ()=> this._ocrSaveSrt());
	  $('ocr-export')?.addEventListener('click', ()=> this._ocrExportSrt());

	  // CC controls
	  $('cc-font')?.addEventListener('input', (e)=>{
		this.ocr.cc.font = Number(e.target.value||18);
		$('cc-font-val').textContent = String(this.ocr.cc.font);
		this._ocrRefreshOsdStyle();
	  });
	  $('cc-pos')?.addEventListener('input', (e)=>{
		this.ocr.cc.bottomPct = Number(e.target.value||4);
		$('cc-pos-val').textContent = String(this.ocr.cc.bottomPct)+'%';
		this._ocrRefreshOsdStyle();
	  });

	  // Delegation list
	  const wrap = $('ocr-list');
	  wrap?.addEventListener('click', (e)=>{
		const btn = e.target.closest?.('[data-play]');
		if (btn){ const idx = Number(btn.getAttribute('data-play')); this._ocrSeekToRow(idx, true); }
	  });
	  wrap?.addEventListener('input', (e)=>{
		const inp = e.target.closest?.('input[data-idx]');
		if (inp){
		  const idx = Number(inp.getAttribute('data-idx'));
		  const row = this.ocr.rows.find(r=>r.index===idx);
		  if (row){
			row.text = inp.value;                       // 1 baris
			row.warn = this._ocrAnalyzeText(row.text);  // refresh warning
			this._ocrRefreshRow(idx);
		  }
		}
	  });
	}
	
	_hasCJK(text){
	  const t = String(text||'');
	  try { return /\p{Script=Han}/u.test(t); }
	  catch(e){ return /[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]/.test(t); }
	}

	/* ========= OCR — loaders ========= */
	async _ocrPromptLoadSrt(){
	  this.initOcrState();
	  const input = document.createElement('input'); input.type='file'; input.accept='.srt';
	  input.onchange = async ()=>{
		const f = input.files?.[0]; if (!f) return;
		const text = await f.text();
		const list = this.parseSRT(text);
		this.ocr.rows = list.map(r=> ({
		  ...r,
		  startS: this._ocrToSec(r.start),
		  endS:   this._ocrToSec(r.end),
		  warn:   this._ocrAnalyzeText(r.text)
		}));
		this._ocrRenderList();
		this.showNotification('SRT OCR loaded','success');
	  };
	  input.click();
	}
	async _ocrPromptLoadVideo(){
	  const input = document.createElement('input'); input.type='file'; input.accept='video/*';
	  input.onchange = ()=>{
		const f = input.files?.[0]; if (!f) return;
		const url = URL.createObjectURL(f);
		const v = document.getElementById('ocr-video');
		if (v){ v.src = url; this.ocr.videoUrl = url; }
	  };
	  input.click();
	}

	/* ========= OCR — warning rules (>=2 baris) ========= */
	_ocrAnalyzeText(text){
	  const warn = [];
	  const t = String(text||'');
	  const lines = t.split('\n');
	  if (lines.length >= 2) warn.push('2+ lines');
	  if (/[A-Za-z]/.test(t)) warn.push('Latin A-Z');
	  if (/[ \t]/.test(t)) warn.push('Spaces');
	  if (/[0-9]/.test(t)) warn.push('Digits');
	  // CJK ratio optional
	  const only = t.replace(/\s/g,'');
	  const cjk = (only.match(/[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]/g)||[]).length;
	  const non = (only.match(/[^\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]/g)||[]).length;
	  const ratio = (cjk + non) ? (cjk/(cjk+non)) : 1;
	  if (ratio < 0.8) warn.push(`Low CJK ${(ratio*100).toFixed(0)}%`);
	  return warn;
	}

	/* ========= OCR — render list (tanpa pagination, search teks saja, input 1 baris) ========= */
	_ocrRenderList(){
	  const oc=this.ocr; if(!oc) return;
	  const wrap=document.getElementById('ocr-list'); if(!wrap) return;
	  const q=(oc.search||'').trim().toLowerCase();

	// filter: Only warnings (China) + Only warnings ID + search (teks saja)
	const rows = (oc.rows||[]).filter(r=>{
	  const txt = String(r.text||'');
	  const hasWarn = (r.warn && r.warn.length);

	  if (oc.onlyWarn && !hasWarn) return false;           // existing: pakai r.warn[] (mode China)
	  if (oc.onlyWarnID && !this._hasCJK(txt)) return false; // NEW: tampilkan hanya baris yang mengandung huruf CJK (untuk input Indonesia)

	  if (!q) return true;
	  return txt.toLowerCase().includes(q);
	});

	  oc.filtered = rows;

	  if (!rows.length){
		wrap.innerHTML = `<div class="p-4 text-gray-300">No rows.</div>`;
		return;
	  }

	  const esc = (s)=> (typeof this._ocrEsc === 'function' ? this._ocrEsc(s) : String(s||''));
	  wrap.innerHTML = rows.map(r=>{
		const oneLine = String(r.text||'').replace(/\n/g,' ');
		// chip compact
		const warnHtml = this._ocrWarnCompact(r.warn||[]);

		return `
		  <div id="ocr-row-${r.index}" class="mb-2 rounded bg-gray-800 p-2">
			<div class="flex items-center gap-2 whitespace-nowrap">
						  <!-- play -->
			  <button data-play="${r.index}"
					  class="ml-2 px-2 py-1 shrink-0 rounded bg-gray-700 hover:bg-gray-600 text-xs">
				<i class="fas fa-play mr-1"></i>Play
			  </button>
			  <!-- #no + time -->
			  <div class="flex items-center gap-2 w-[210px] shrink-0">
				<div class="font-mono text-xs bg-gray-900/60 px-2 py-1 rounded">#${r.index}</div>
				<div class="font-mono text-xs text-gray-300">${esc(r.start)} → ${esc(r.end)}</div>
			  </div>

			  <!-- teks (1 baris, fleksibel) -->
			  <input data-idx="${r.index}" type="text"
					 class="flex-1 min-w-0 h-9 bg-gray-900 rounded px-2 outline-none"
					 value="${esc(oneLine)}" />

			  <!-- warning (compact, dibatasi supaya tidak mendorong tombol) -->
			  <div id="ocr-warn-${r.index}"
				   class="ml-2 shrink-0 max-w-[32%] overflow-hidden">
				${warnHtml}
			  </div>


			</div>
		  </div>`;
	  }).join('');
	}


	_ocrRefreshRow(idx){
	  const row = this.ocr.rows.find(r=>r.index===idx); if (!row) return;
	  const box = document.getElementById(`ocr-warn-${idx}`); if (!box) return;
	  box.innerHTML = this._ocrWarnCompact(row.warn||[]);
	}


	/* ========= OCR — OSD CC style & text ========= */
	_ocrRefreshOsdStyle(){
	  const osd = document.getElementById('ocr-osd'); if (!osd) return;
	  osd.style.fontSize = `${this.ocr.cc.font}px`;
	  osd.style.bottom   = `${this.ocr.cc.bottomPct}%`;
	  osd.style.width    = `${this.ocr.cc.widthPct}%`; // <= lebar kontrol
	  osd.style.left     = '50%';
	  osd.style.transform= 'translateX(-50%)';
	  osd.style.whiteSpace = 'normal';
	  osd.style.wordBreak  = 'break-word';
	}

	_ocrUpdateOsd(text){
	  const osd = document.getElementById('ocr-osd'); if (!osd) return;
	  osd.innerHTML = this._ocrEsc(String(text||'')).replace(/\n/g,'<br>');
	}

	// buat chip warning compact: tampilkan 2 pertama + "+N"
	_ocrWarnCompact(warns){
	  const esc = (s)=> (typeof this._ocrEsc === 'function' ? this._ocrEsc(s) : String(s||''));
	  const list = Array.isArray(warns) ? warns : [];
	  const title = esc(list.join(', '));
	  const shown = list.slice(0, 2);
	  const chips = shown.map(w => (
		`<span class="inline-flex items-center px-2 py-0.5 rounded bg-red-600/20 text-red-300 text-xs">⚠ ${esc(w)}</span>`
	  )).join('');
	  const more = (list.length > 2)
		? `<span class="inline-flex items-center px-2 py-0.5 rounded bg-red-600/20 text-red-300 text-xs cursor-default" title="${title}">+${list.length-2}</span>`
		: '';
	  // bungkus satu kontainer agar bisa diupdate parsial
	  return `<div class="flex items-center gap-1" title="${title}">${chips}${more}</div>`;
	}

	/* ========= OCR — playback (auto-follow list only + update OSD) ========= */
	_ocrSeekToRow(idx, play=false){
	  const oc=this.ocr; const v=document.getElementById('ocr-video');
	  const r=(oc.rows||[]).find(x=>x.index===idx); if(!r||!v) return;
	  v.currentTime = (r.startS ?? this._ocrToSec(r.start)) + 0.001;
	  this._ocrHighlight(idx);
	  this._ocrScrollTo(idx);
	  this._ocrUpdateOsd(r.text);
	  if (play){ oc.playUntil = (r.endS ?? this._ocrToSec(r.end)); v.play(); }
	}
	_ocrOnVideoTime(t){
	  const oc=this.ocr; if(!oc?.rows?.length) return;
	  const r = oc.rows.find(x=> t>=(x.startS??this._ocrToSec(x.start)) && t < (x.endS??this._ocrToSec(x.end)) );
	  if (r){
		this._ocrHighlight(r.index);
		if (oc.filtered?.some?.(x=>x.index===r.index)) this._ocrScrollTo(r.index); // follow list only
		this._ocrUpdateOsd(r.text);
	  } else {
		this._ocrUpdateOsd('');
	  }
	  if (oc.playUntil != null && t >= oc.playUntil - 0.02){
		const v=document.getElementById('ocr-video'); v?.pause(); oc.playUntil = null;
	  }
	}
	_ocrScrollTo(idx){
	  const wrap=document.getElementById('ocr-list');
	  const row=document.getElementById(`ocr-row-${idx}`);
	  if(!wrap||!row) return;
	  const top=row.offsetTop - wrap.offsetTop;
	  wrap.scrollTo({ top: Math.max(0, top - wrap.clientHeight*0.3), behavior:'smooth' });
	}
	_ocrHighlight(idx){
	  const oc=this.ocr; if(!oc) return;
	  if (oc.liveIndex===idx) return;
	  if (oc.liveIndex!=null){
		const prev=document.getElementById(`ocr-row-${oc.liveIndex}`);
		prev?.classList.remove('ring-2','ring-blue-400');
	  }
	  const now=document.getElementById(`ocr-row-${idx}`);
	  now?.classList.add('ring-2','ring-blue-400');
	  oc.liveIndex = idx;
	}

	/* ========= OCR — save/export (download) ========= */
	_ocrBuildSrt(renumber = true){
	  // Ambil hanya baris yang masih punya teks (non-kosong setelah trim)
	  const rows = (this.ocr.rows || []).filter(r => (String(r.text||'').trim().length > 0));

	  let n = 1;
	  return rows.map(r => {
		const idx = renumber ? (n++) : r.index;
		const txt = String(r.text || '');
		return `${idx}\n${r.start} --> ${r.end}\n${txt}\n`;
	  }).join('\n');
	}
	_ocrSaveSrt(){
	  const srt = this._ocrBuildSrt(true);  // renumber
	  this._ocrDownload('ocr_cleaned.srt', srt);
	  this.showNotification('Saved ocr_cleaned.srt','success');
	}
	_ocrExportSrt(){
	  const ts=new Date(), pad=n=>String(n).padStart(2,'0');
	  const name=`ocr_cleaned_${ts.getFullYear()}${pad(ts.getMonth()+1)}${pad(ts.getDate())}_${pad(ts.getHours())}${pad(ts.getMinutes())}.srt`;
	  const srt=this._ocrBuildSrt(true);    // renumber
	  this._ocrDownload(name, srt);
	  this.showNotification(`Exported ${name}`,'success');
	}

	_ocrDownload(name, text){
	  const blob=new Blob([text], {type:'text/plain'}); const url=URL.createObjectURL(blob);
	  const a=document.createElement('a'); a.href=url; a.download=name; a.click();
	  setTimeout(()=> URL.revokeObjectURL(url), 1000);
	}

	// === TAB: REVIEW & EXPORT (CapCut Project) ===
// === TAB: REVIEW & EXPORT (Tahap 1,2,3) ===
async loadExportTab() {
  const tab = document.getElementById('export');
  if (!tab) return;

  // ensure some state on 'this'
  this.exportBusy = this.exportBusy || false;
  this.export = this.export || { sessionId: this.currentSessionId || '', rows: [] };
  this.exMap = this.exMap || { global_nudge_ms: 0, row_nudges: {} };

  // ---------- MARKUP ----------
  tab.innerHTML = `
    <div class="p-4 bg-gray-800 rounded-lg">
      <h2 class="text-2xl font-bold mb-4 text-blue-400">
        <i class="fas fa-cloud-upload-alt mr-2"></i>Review & Export
      </h2>

      <!-- Session -->
      <div class="flex items-center gap-2 mb-3">
        <select id="ex-session-list" class="px-2 py-1 bg-gray-700 border border-gray-600 rounded w-[260px] text-white">
          <option value="">— pilih session —</option>
        </select>
        <button id="ex-load-session" class="btn btn-primary btn-sm">Load Session</button>
        <span id="ex-session-label" class="ml-2 text-gray-300"></span>
      </div>

      <!-- Tahap-1: Import -->
      <div class="bg-gray-700 rounded p-3 mb-4">
        <div class="text-sm text-gray-200 font-semibold mb-2">CapCut TTS Import (Project)</div>
        <div class="flex flex-wrap items-center gap-2">
          <span class="text-xs text-gray-300">Project dir</span>
          <input id="ex-project-dir" type="text" class="px-2 py-1 bg-gray-800 border border-gray-600 rounded w-[620px] text-white"
                 placeholder="C:\\Users\\...\\CapCut\\User Data\\Projects\\com.lveditor.draft\\1022" />
          <button id="ex-imp-project" class="btn btn-primary btn-sm">Import (project)</button>
        </div>

        <div class="mt-2 flex flex-wrap items-center gap-2">
          <button id="ex-detect" class="btn btn-slate btn-xs">Detect offset</button>
          <button id="ex-remap"  class="btn btn-slate btn-xs">Remap strict</button>

          <span class="text-xs text-gray-300">offset</span>
          <input id="ex-offset" type="number" value="0" class="w-24 px-2 py-1 bg-gray-800 border border-gray-600 rounded text-white" />
          <span id="ex-effective" class="text-xs text-emerald-300">effective offset: +0ms</span>

          <span class="text-xs text-gray-300 ml-3">tol (ms)</span>
          <input id="ex-tolerance" type="number" value="200" class="w-24 px-2 py-1 bg-gray-800 border border-gray-600 rounded text-white" />

          <div class="flex items-center gap-2 ml-2">
            <button id="ex-global-minus" class="btn btn-xs">-50ms (All)</button>
            <span id="ex-global-val" class="text-xs px-2 py-0.5 rounded bg-zinc-700">+0ms</span>
            <button id="ex-global-plus" class="btn btn-xs">+50ms (All)</button>
          </div>

          <div class="flex items-center gap-2 ml-4">
            <button id="ex-reset-time" class="btn btn-danger btn-xs">Reset Time</button>
          </div>
        </div>
      </div>

      <div class="grid grid-cols-3 gap-4">
        <div class="col-span-1">
          <video id="ex-video" class="w-full rounded bg-black" controls playsinline></video>
          <audio id="ex-audio" class="mt-2 w-full" controls></audio>

          <!-- Play All Controls -->
          <div class="mt-2 flex gap-2">
            <button id="ex-play-all" class="btn btn-primary btn-sm">Play All</button>
            <button id="ex-stop-all" class="btn btn-slate btn-sm">Stop</button>
            <span id="ex-prep-hint" class="text-xs text-gray-300 hidden">menyiapkan audio…</span>
          </div>

          <!-- Tahap-3: Export -->
          <div class="mt-4 bg-gray-700 p-3 rounded">
            <div class="text-sm text-gray-200 font-semibold mb-2">Export</div>
            <label class="text-xs text-gray-300 flex items-center gap-2 mb-1">
              <input id="ex-center" type="checkbox" checked class="mr-1"> Center cut / ducking (untuk export)
            </label>
            <div class="grid grid-cols-2 gap-2 mb-2">
              <label class="text-xs text-gray-300 flex items-center gap-2">
                TTS vol
                <input id="ex-tts-vol" type="number" step="0.05" value="1.0" class="w-24 px-2 py-1 bg-gray-800 border border-gray-600 rounded text-white" />
              </label>
              <label class="text-xs text-gray-300 flex items-center gap-2">
                BG vol
                <input id="ex-bg-vol" type="number" step="0.05" value="0.8" class="w-24 px-2 py-1 bg-gray-800 border border-gray-600 rounded text-white" />
              </label>
              <label class="text-xs text-gray-300 flex items-center gap-2">
                Max atempo
                <input id="ex-max-atempo" type="number" step="0.1" value="2.8" class="w-24 px-2 py-1 bg-gray-800 border border-gray-600 rounded text-white" />
              </label>
              <label class="text-xs text-gray-300 flex items-center gap-2">
                MIN atempo
                <input id="ex-min-atempo" type="number" step="0.05" value="1.4" class="w-24 px-2 py-1 bg-gray-800 border border-gray-600 rounded text-white" />
              </label>
              <label class="text-xs text-gray-300 flex items-center gap-2">
                Base Tempo
                <input id="ex-base-tempo" type="number" step="0.05" value="1.4" class="w-24 px-2 py-1 bg-gray-800 border border-gray-600 rounded text-white" />
              </label>
              <label class="text-xs text-gray-300 flex items-center gap-2">
                Audio bitrate
                <input id="ex-audio-br" type="text" value="128k" class="w-24 px-2 py-1 bg-gray-800 border border-gray-600 rounded text-white" />
              </label>
			  
				<!-- ⬇⬇⬇ BARU: chunk_size -->
				<label class="text-xs text-gray-300 flex items-center gap-2 col-span-2">
				  chunk_size
				  <input id="ex-chunk-size" type="number" min="1" max="250" value="200"
					class="w-24 px-2 py-1 bg-gray-800 border border-gray-600 rounded text-white" />
				  <span class="text-[10px] text-gray-400">
					besar batch TTS per proses ffmpeg (max 200 aman di Windows)
				  </span>
				</label>
				<!-- ⬆⬆⬆ BARU -->
			  
            </div>
            <label class="text-xs text-gray-300 flex items-center gap-2 mb-2">
              BGM (opsional, rel/abs path)
              <input id="ex-bgm" type="text" class="w-full px-2 py-1 bg-gray-800 border border-gray-600 rounded text-white" placeholder="paste url file instrument.wav di workspaces">
            </label>
            <div class="flex gap-2 flex-wrap items-center">
              <button id="ex-build" class="btn btn-primary btn-sm">Export MP4</button>
              <a id="ex-download" class="hidden btn btn-slate btn-sm" href="#" target="_blank">Download</a>
              <!-- tombol copy instrument.wav → BGM akan disisipkan via JS -->
            </div>
          </div>
        </div>

        <div class="col-span-2">
          <div id="ex-rows" class="h-[100vh] overflow-auto border border-gray-700 rounded bg-gray-900"></div>
        </div>
      </div>
    </div>
  `;

  // ---------- UTIL / SHORTCUT ----------
  const $   = id => document.getElementById(id);
  const sid = () => this.export.sessionId;
  const fmtMs = v => `${v >= 0 ? '+' : ''}${v}ms`;

  const updateEffective = () => {
    const eff = Number($('ex-offset').value || 0) + Number(this.exMap?.global_nudge_ms || 0);
    const lab = $('ex-effective');
    if (lab) lab.textContent = `effective offset: ${fmtMs(eff)}`;
  };

  const setVideoSrc = s => {
    const v = $('ex-video');
    v.src = s ? `/api/session/${encodeURIComponent(s)}/video` : '';
    v.onerror = () => this.showNotification('Preview gagal: video tidak ditemukan / tidak didukung.', 'error');
  };

  const _toSid = (x) => {
    if (x == null) return '';
    if (typeof x === 'string' || typeof x === 'number') return String(x);
    if (typeof x === 'object') {
        if (x.session_id) return String(x.session_id);
        if (x.id)         return String(x.id);
        if (x.name)       return String(x.name);
        if (x.value)      return String(x.value);
        const ks = Object.keys(x);
        if (ks.length === 1) return String(x[ks[0]]);
        return JSON.stringify(x);
    }
    return String(x);
  };

  // ---------- API: MAP OFFSET / NUDGE ----------
  const exLoadMap = async () => {
    if (!sid()) return;
    try {
      const r = await fetch(`/api/session/${encodeURIComponent(sid())}/capcut/map`);
      if (!r.ok) return;
      const j = await r.json();
      this.exMap = j || { global_nudge_ms: 0, row_nudges: {} };
      $('ex-global-val').textContent = fmtMs(this.exMap.global_nudge_ms || 0);
      updateEffective();
    } catch {}
  };

  // ---------- PREVIEW AUDIO / PLAYBACK STATE ----------
  // state untuk WebAudio preview
  let _waCtx     = null;
  let _waGain    = null;
  let _waSources = [];
  let _waCleanupVideo = null;

  // bikin preview audio gabungan pitch-normal dari backend, dan simpan tempo map per-row
  const exBuildPreviewAudio = async () => {
    if (!sid()) throw new Error('Load Session dulu');

    const payload = {
      base_tempo:   Number($('ex-base-tempo')?.value || 1.4),
      max_atempo:   Number($('ex-max-atempo')?.value || 2.8),
      tts_vol:      Number($('ex-tts-vol')?.value || 1.0),
      bg_vol:       Number($('ex-bg-vol')?.value  || 0.8),
      center_cut:   !!$('ex-center')?.checked,
      audio_bitrate:'128k',
      guard_ms:     15,
      safety_ms:    80,
    };

    const res = await fetch(`/api/session/${encodeURIComponent(sid())}/export/preview-audio`, {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(payload),
    });

    const text = await res.text();
    if (!res.ok) throw new Error(text || 'Build preview audio gagal');

    let j = {};
    try { j = JSON.parse(text); } catch {}

    if (!j?.file) throw new Error('Preview tidak mengembalikan file.');

    // ambil tempo_map, isinya { "<row_index>": { url, eff_start_ms, ... }, ... }
    let tempoMap = {};
    if (j.tempo_map) {
      try {
        const tmRes = await fetch(j.tempo_map, { cache: 'no-store' });
        if (tmRes.ok) tempoMap = await tmRes.json();
      } catch {}
    }

    // simpan global agar bisa dipakai playSegment()
    window._exPreview = { url: j.file, tempoMap };
    window._exTempoMap = tempoMap; // kompat

    return j.file; // URL preview full mix
  };

  // stop semua audio & balikin video
  const stopPlayAll = () => {
    try { _waSources.forEach(s => { try { s.stop(0); } catch {} }); } catch {}
    _waSources = [];

    if (_waCtx && _waCtx.state === 'running') {
      _waCtx.suspend().catch(()=>{});
    }

    const v = $('ex-video');
    try { v.pause(); } catch {}

    if (_waCleanupVideo) {
      try { _waCleanupVideo(); } catch {}
      _waCleanupVideo = null;
    }

    $('ex-play-all')?.removeAttribute('disabled');
    $('ex-prep-hint')?.classList.add('hidden');
  };

  // play satu subtitle baris (pakai entry dari tempo_map)
  const playSegment = async (rowIdx) => {
    if (!sid()) { this.showNotification('Load Session dulu', 'error'); return; }

    // pastikan tempoMap ready
    const tempoMapHave =
      (window._exPreview?.tempoMap && window._exPreview.tempoMap[String(rowIdx)]) ||
      (window._exTempoMap && window._exTempoMap[String(rowIdx)]);

    if (!tempoMapHave) {
      try { await exBuildPreviewAudio(); }
      catch (e) {
        this.showNotification(String(e.message || e), 'error');
        return;
      }
    }

    const info =
      (window._exPreview?.tempoMap && window._exPreview.tempoMap[String(rowIdx)]) ||
      (window._exTempoMap && window._exTempoMap[String(rowIdx)]);

    if (!info?.url) {
      this.showNotification('URL segmen tidak ditemukan.', 'error');
      return;
    }

    // siapkan AudioContext kalau belum
    if (!_waCtx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      _waCtx = new AC({ sampleRate: 48000 });
      _waGain = _waCtx.createGain();
      _waGain.gain.value = Math.max(0, Math.min(1, Number($('ex-tts-vol')?.value || 1.0)));
      _waGain.connect(_waCtx.destination);
    } else if (_waCtx.state === 'suspended') {
      try { await _waCtx.resume(); } catch {}
    }

    // fetch & decode audio segmen
    try {
      const r = await fetch(info.url, { cache: 'no-store' });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const ab = await r.arrayBuffer();
      const buf = await _waCtx.decodeAudioData(ab);

      // matikan source lama
      try { _waSources.forEach(s => { try { s.stop(0); } catch {} }); } catch {}
      _waSources = [];

      // sinkronkan video ke posisi mulai dialog ini
      const v = $('ex-video');
      if (v && info.eff_start_ms != null) {
        v.muted = true;
        v.volume = 0;
        v.currentTime = info.eff_start_ms / 1000;
        try { await v.play(); } catch {}
        _waCleanupVideo = () => {
          v.pause();
          v.muted = false;
          v.volume = 1.0;
        };
      }

      const src = _waCtx.createBufferSource();
      src.buffer = buf;
      src.connect(_waGain);
      src.start(_waCtx.currentTime);
      _waSources.push(src);

      setTimeout(() => {
        stopPlayAll();
      }, Math.max(0, buf.duration * 1000) + 150);

    } catch (e) {
      this.showNotification('Gagal memutar segmen.', 'error');
    }
  };

  // play semua (preview-mix gabungan) sekali jalan
  const playAll = async () => {
    if (!sid()) {
      this.showNotification('Load Session dulu', 'error');
      return;
    }

    $('ex-play-all')?.setAttribute('disabled', 'true');
    $('ex-prep-hint')?.classList.remove('hidden');

    // siapkan AudioContext
    if (!_waCtx) {
      const AC = window.AudioContext || window.webkitAudioContext;
      _waCtx = new AC({ sampleRate: 48000 });
      _waGain = _waCtx.createGain();
      _waGain.gain.value = Math.max(0, Math.min(1, Number($('ex-tts-vol')?.value || 1.0)));
      _waGain.connect(_waCtx.destination);

      // live update volume TTS
      $('ex-tts-vol')?.addEventListener('input', () => {
        if (_waGain) {
          _waGain.gain.value = Math.max(0, Math.min(1, Number($('ex-tts-vol')?.value || 1.0)));
        }
      });
    } else if (_waCtx.state === 'suspended') {
      try { await _waCtx.resume(); } catch {}
    }

    // pastikan preview_mix ada
    let previewUrl = '';
    try {
      previewUrl = await exBuildPreviewAudio();
    } catch (e) {
      $('ex-prep-hint')?.classList.add('hidden');
      $('ex-play-all')?.removeAttribute('disabled');
      this.showNotification(String(e.message || e), 'error');
      return;
    }

    // ambil & decode full preview_mix
    let buf = null;
    try {
      const r = await fetch(previewUrl, { cache: 'no-store' });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      const ab = await r.arrayBuffer();
      buf = await _waCtx.decodeAudioData(ab);
    } catch (e) {
      $('ex-prep-hint')?.classList.add('hidden');
      $('ex-play-all')?.removeAttribute('disabled');
      this.showNotification('Decode preview audio gagal (file mungkin kosong).', 'error');
      return;
    }

    // siapkan video global muting
    const v = $('ex-video');
    const prevMuted = v?.muted;
    const prevVol   = v?.volume;
    if (v) {
      v.muted = true;
      v.volume = 0;
      v.currentTime = 0;
      _waCleanupVideo = () => {
        v.pause();
        v.muted = prevMuted;
        v.volume = prevVol;
      };
      try { await v.play(); } catch {}
    }

    try {
      // stop sumber lama
      try { _waSources.forEach(s => { try { s.stop(0); } catch {} }); } catch {}
      _waSources = [];

      const src = _waCtx.createBufferSource();
      src.buffer = buf;
      src.connect(_waGain);
      src.start(_waCtx.currentTime);
      _waSources.push(src);

      const durMs = Math.max(0, (buf.duration || 0) * 1000);
      setTimeout(() => stopPlayAll(), durMs + 200);
    } catch (e) {
      this.showNotification('Gagal memutar preview audio.', 'error');
      stopPlayAll();
      return;
    } finally {
      $('ex-prep-hint')?.classList.add('hidden');
      $('ex-play-all')?.removeAttribute('disabled');
    }
  };

  // ---------- NUDGE ----------
  const nudgeRow = async (idx, delta) => {
    try {
      const r = await fetch(`/api/session/${encodeURIComponent(sid())}/capcut/nudge_row`, {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ index: Number(idx), delta_ms: Number(delta) })
      });
      const j = await r.json();
      if (!r.ok) throw new Error(j?.detail || 'Nudge gagal');
      await exLoadMap();
      const cur = (this.exMap?.row_nudges || {})[String(idx)] || 0;
      const span = document.getElementById(`ex-nudge-val-${idx}`);
      if (span) span.textContent = fmtMs(cur);
    } catch (e) {
      this.showNotification(String(e), 'error');
    }
  };

  const nudgeAll = async (delta) => {
    try {
      const r = await fetch(`/api/session/${encodeURIComponent(sid())}/capcut/nudge_all`, {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({ delta_ms: Number(delta) })
      });
      const j = await r.json();
      if (!r.ok) throw new Error(j?.detail || 'Nudge all gagal');
      await exLoadMap();
      await reloadRows();
    } catch (e) {
      this.showNotification(String(e), 'error');
    }
  };

  // Reset semua timing (global+row nudges)
  const resetTime = async () => {
    if (!sid()) return;
    try {
      // direct reset
      let r = await fetch(`/api/session/${encodeURIComponent(sid())}/capcut/reset_nudges`, {
        method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify({})
      });
      if (!r.ok) {
        // fallback manual
        await exLoadMap();
        const g = Number(this.exMap?.global_nudge_ms || 0);
        if (g) {
          await fetch(`/api/session/${encodeURIComponent(sid())}/capcut/nudge_all`, {
            method:'POST', headers:{'Content-Type':'application/json'},
            body: JSON.stringify({ delta_ms: -g })
          });
        }
        const rn = this.exMap?.row_nudges || {};
        const keys = Object.keys(rn);
        for (const k of keys) {
          const v = Number(rn[k] || 0);
          if (!v) continue;
          await fetch(`/api/session/${encodeURIComponent(sid())}/capcut/nudge_row`, {
            method:'POST', headers:{'Content-Type':'application/json'},
            body: JSON.stringify({ index: Number(k), delta_ms: -v })
          });
        }
      }
      await exLoadMap();
      await reloadRows();
      this.showNotification('Timing direset ke kondisi asli.', 'success');
    } catch (e) {
      this.showNotification('Reset gagal.', 'error');
    }
  };

  // ---------- SESSION / ROW DATA ----------
  const fillSessions = async () => {
    const sel = $('ex-session-list');
    sel.innerHTML = `<option value="">— pilih session —</option>`;
    try {
      const r = await fetch('/api/sessions');
      const j = await r.json();
      const arr = Array.isArray(j) ? j : (j.sessions || j.data || j.items || []);
      const seen = new Set();
      (arr || []).forEach(item => {
        const id = _toSid(item).trim();
        if (!id || seen.has(id)) return;
        seen.add(id);
        const op = document.createElement('option');
        op.value = id;
        op.textContent = id;
        sel.appendChild(op);
      });
      if (this.currentSessionId && seen.has(this.currentSessionId)) {
        sel.value = this.currentSessionId;
      }
    } catch (e) {
      console.error('sessions payload:', e);
      this.showNotification('Gagal memuat daftar session.', 'error');
    }
  };

  const loadSession = async (s) => {
    if (!s) { this.showNotification('Pilih session terlebih dahulu.', 'warning'); return; }
    this.export.sessionId = s;

    const lab = $('ex-session-label');
    if (lab) lab.textContent = String(s);

    setVideoSrc(s);

    try {
      const r = await fetch(`/api/session/${encodeURIComponent(s)}/review`);
      const j = await r.json();
      this.export.rows = j.rows || [];
      await exLoadMap();
      renderRows();
    } catch (e) {
      this.showNotification('Gagal memuat data session.', 'error');
    }
  };

  const reloadRows = async () => {
    try {
      const r = await fetch(`/api/session/${encodeURIComponent(sid())}/review`);
      const j = await r.json();
      this.export.rows = j.rows || [];
      renderRows();
    } catch {
      this.showNotification('Gagal memuat data session', 'error');
    }
  };

  const msToSrt = (ms) => {
    const z = m => String(m).padStart(2,'0');
    const ms3 = String(ms%1000).padStart(3,'0');
    const s = Math.floor(ms/1000);
    const h = Math.floor(s/3600), m = Math.floor((s%3600)/60), ss = s%60;
    return `${z(h)}:${z(m)}:${z(ss)},${ms3}`;
  };

  // render daftar baris subtitle + tombol Play + nudge
  const renderRows = () => {
    const box = $('ex-rows');
    const rows = this.export.rows || [];
    if (!rows.length) {
      box.innerHTML = `<div class="p-3 text-gray-400">No rows.</div>`;
      return;
    }

    const rnudges = (this.exMap?.row_nudges) || {};
    box.innerHTML = rows.map(r => {
      const hasTTS = !!r.tts_path;
      const nudge = Number(rnudges[String(r.index)] || 0);
      return `
        <div class="flex items-center justify-between px-3 py-2 border-b border-gray-800">
          <div class="text-xs text-gray-400 w-[160px]">
            <div>${msToSrt(r.start_ms)}</div>
            <div>${msToSrt(r.end_ms)}</div>
          </div>
          <div class="text-xs flex-1">
            ${r.gender ? `<span class="px-2 py-0.5 rounded bg-gray-700 text-gray-200 mr-2">${r.gender}</span>` : ''}
            <span class="text-gray-100">${this.escapeHtml(r.text||'')}</span>
          </div>
          <div class="ml-2 flex items-center gap-2">
            ${hasTTS ? `<button class="btn btn-primary btn-xs" data-play="${r.index}">Play</button>`
                      : `<span class="text-xs text-gray-500">No TTS</span>`}
            <div class="flex items-center gap-1" data-nudge="${r.index}">
              <button class="btn btn-xs" data-nudge-delta="-50">-50ms</button>
              <span id="ex-nudge-val-${r.index}" class="text-xs px-2 py-0.5 rounded bg-zinc-700">${fmtMs(nudge)}</span>
              <button class="btn btn-xs" data-nudge-delta="+50">+50ms</button>
            </div>
          </div>
        </div>
      `;
    }).join('');
  };

  // ---------- EXPORT UI HELPERS ----------
  const lockExportUI = (isLocked, msg = 'Export sedang berjalan... mohon tunggu') => {
    const tabEl = document.getElementById('export');
    if (!tabEl) return;

    // disable semua tombol di dalam tab export
    tabEl.querySelectorAll('button').forEach(btn => {
      if (isLocked) {
        btn.setAttribute('disabled', 'true');
        btn.classList.add('opacity-50', 'cursor-not-allowed');
      } else {
        btn.removeAttribute('disabled');
        btn.classList.remove('opacity-50', 'cursor-not-allowed');
      }
    });

    // badge kecil di samping tombol Export
    const buildBtn = $('ex-build');
    if (!buildBtn) return;

    let badge = document.getElementById('ex-progress');
    if (isLocked) {
      if (!badge) {
        badge = document.createElement('span');
        badge.id = 'ex-progress';
        badge.className = 'ml-2 inline-flex items-center text-xs text-yellow-400';
        badge.innerHTML = `
          <span class="loading-spinner w-4 h-4 border-2 border-t-transparent border-yellow-400 rounded-full inline-block mr-2"></span>
          <span class="ex-progress-text">${msg}</span>
        `;
        buildBtn.parentElement.appendChild(badge);
      } else {
        badge.classList.remove('hidden');
        const t = badge.querySelector('.ex-progress-text');
        if (t) t.textContent = msg;
      }
    } else {
      if (badge) {
        badge.classList.add('hidden');
      }
    }
  };

  // popup progress pojok kanan bawah
  const ensureExportModal = () => {
    let box = document.getElementById('ex-progress-modal');
    if (box) return box;

    box = document.createElement('div');
    box.id = 'ex-progress-modal';
    box.className = [
      'fixed bottom-4 right-4 z-50 w-72',
      'bg-gray-800 border border-gray-700 rounded-xl shadow-xl',
      'text-white text-sm'
    ].join(' ');

    box.innerHTML = `
      <div class="p-4 space-y-3">
        <div class="flex items-start justify-between">
          <div>
            <div class="font-semibold text-white text-sm">Export Progress</div>
            <div id="ex-progress-status" class="text-xs text-gray-300">
              Persiapan…
            </div>
          </div>
          <button id="ex-progress-close"
                  class="text-gray-400 hover:text-white text-xs">
            <i class="fas fa-times"></i>
          </button>
        </div>

        <div class="w-full bg-gray-700 rounded h-2 overflow-hidden">
          <div id="ex-progress-bar"
               class="bg-green-500 h-2"
               style="width:0%"></div>
        </div>

        <div class="flex justify-between text-[11px] text-gray-400">
          <span id="ex-progress-pct">0%</span>
          <a id="ex-progress-open"
             class="hidden text-blue-400 hover:underline"
             target="_blank" rel="noopener">
             Open MP4
          </a>
        </div>
      </div>
    `;

    document.body.appendChild(box);

    box.querySelector('#ex-progress-close').onclick = () => {
      box.remove();
    };

    return box;
  };

  const updateExportModal = (pct, status, downloadUrl) => {
    const box   = ensureExportModal();
    const bar   = box.querySelector('#ex-progress-bar');
    const pctEl = box.querySelector('#ex-progress-pct');
    const stEl  = box.querySelector('#ex-progress-status');
    const openA = box.querySelector('#ex-progress-open');

    if (typeof pct === 'number' && bar) {
      bar.style.width = `${pct}%`;
    }
    if (typeof pct === 'number' && pctEl) {
      pctEl.textContent = `${pct}%`;
    }
    if (status && stEl) {
      stEl.textContent = status;
    }
    if (downloadUrl) {
      openA.classList.remove('hidden');
      openA.href = downloadUrl;
    }
  };

  // ---------- CENTER CUT <-> BGM UI SYNC ----------
  const setupCenterCutAndBgmUI = () => {
    const bgmInput  = $('ex-bgm');
    const centerChk = $('ex-center');
    if (!bgmInput || !centerChk) return;

    const showMode = () => {
      const hasBgm    = (bgmInput.value || '').trim() !== '';
      const useCenter = centerChk.checked && !hasBgm;
      let el = document.getElementById('ex-mode');
      if (!el) {
        el = document.createElement('span');
        el.id = 'ex-mode';
        el.className = 'ml-2 text-xs px-2 py-1 rounded bg-gray-700/40';
        (centerChk.parentElement || centerChk).appendChild(el);
      }
      el.textContent = useCenter ? 'Mode: Center-cut' : (hasBgm ? 'Mode: BGM' : 'Mode: Default');
    };

    const syncCenterCutState = () => {
      const hasBgm = (bgmInput.value || '').trim() !== '';
      centerChk.disabled = hasBgm;
      if (hasBgm) {
        centerChk.checked = false;
        centerChk.title = 'Center-cut hanya dipakai jika BGM kosong';
      } else {
        centerChk.checked = true;
        centerChk.title = '';
      }
      showMode();
    };

    // tombol kecil "Copy instrument.wav → BGM"
    if (!document.getElementById('ex-copy-instrument')) {
      const btn = document.createElement('button');
      btn.type = 'button';
      btn.id = 'ex-copy-instrument';
      btn.textContent = 'Copy instrument.wav → BGM';
      btn.className = 'ml-2 px-2 py-1 text-xs rounded bg-gray-700 hover:bg-gray-600';
      btn.onclick = async () => {
        const val = 'instrument.wav'; // relatif workspace
        bgmInput.value = val;
        bgmInput.dispatchEvent(new Event('input', { bubbles: true }));
        try {
          await navigator.clipboard.writeText(val);
          this.showNotification?.('instrument.wav disalin & diisi ke BGM', 'success');
        } catch {
          this.showNotification?.('instrument.wav diisi ke BGM (clipboard mungkin diblokir)', 'info');
        }
        showMode();
      };
      // sisipin setelah container tombol export
      const buildParent = $('ex-build')?.parentElement;
      if (buildParent) {
        buildParent.appendChild(btn);
      } else {
        (centerChk.parentElement || centerChk).insertAdjacentElement('afterend', btn);
      }
    }

    bgmInput.addEventListener('input', syncCenterCutState);
    centerChk.addEventListener('change', syncCenterCutState);

    // init pertama kali
    syncCenterCutState();
    showMode();
  };

  // jalankan sinkron UI center-cut/BGM sekarang
  setupCenterCutAndBgmUI();
// validasi live untuk chunk_size
(() => {
  const cs = document.getElementById('ex-chunk-size');
  if (!cs) return;
  cs.addEventListener('input', () => {
    let v = Number(cs.value || 300);
    if (v > 300) {
      cs.classList.add('border-red-500', 'text-red-400');
    } else if (v <= 0) {
      cs.classList.add('border-red-500', 'text-red-400');
    } else {
      cs.classList.remove('border-red-500', 'text-red-400');
    }
  });
})();

  // ---------- EVENT LISTENERS ----------

  // load session button
  $('ex-load-session').onclick = async () => {
    await loadSession($('ex-session-list').value);
    // auto isi BGM instrument.wav kalau kosong
    const bgm = $('ex-bgm');
    if (bgm && !bgm.value.trim()) {
      bgm.value = 'instrument.wav';
      bgm.dispatchEvent(new Event('input', { bubbles: true }));
    }
  };

  // import CapCut project
  $('ex-imp-project').onclick = async () => {
    if (!sid()) return this.showNotification('Load Session dulu', 'error');
    const prj = $('ex-project-dir').value.trim();
    if (!prj) return this.showNotification('Isi Project dir dulu', 'warning');
    const r = await fetch(`/api/session/${encodeURIComponent(sid())}/capcut/import`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({ project_dir: prj })
    });
    const t = await r.text();
    if (!r.ok) return this.showNotification(`Import gagal: ${t}`, 'error');
    this.showNotification('Import OK', 'success');
    await exLoadMap();
    await reloadRows();
  };

  // detect offset
  $('ex-detect').onclick = async () => {
    const r = await fetch(`/api/session/${encodeURIComponent(sid())}/capcut/detect_offset`, { method: 'POST' });
    const j = await r.json().catch(()=>({}));
    if (!r.ok) return this.showNotification(j?.detail || 'Gagal mendeteksi offset', 'error');
    $('ex-offset').value = j.suggested_offset_ms ?? 0;
    updateEffective();
    this.showNotification(`Suggested offset: ${j.suggested_offset_ms} ms (pairs: ${j.pairs_used})`, 'success');
  };

  // remap strict
  $('ex-remap').onclick = async () => {
    if (!$('ex-tolerance').value) $('ex-tolerance').value = 200;
    const bodyRemap = {
      global_offset_ms: Number($('ex-offset').value || 0),
      tolerance_ms: Number($('ex-tolerance').value || 200),
    };
    const r = await fetch(`/api/session/${encodeURIComponent(sid())}/capcut/remap`, {
      method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(bodyRemap)
    });
    const j = await r.json().catch(()=>({}));
    if (!r.ok) return this.showNotification(j?.detail || JSON.stringify(j), 'error');
    this.showNotification(`Remap: matched ${j.matched}/${j.total_rows}`, 'success');
    await exLoadMap();
    await reloadRows();
  };

  // global nudge +/-50ms
  $('ex-global-minus').onclick = () => nudgeAll(-50);
  $('ex-global-plus').onclick  = () => nudgeAll(+50);

  // reset time
  $('ex-reset-time').onclick = () => resetTime();

  // offset manual input -> update effective label
  $('ex-offset').addEventListener('input', updateEffective);

  // click list rows: Play segmen / nudge row
  document.getElementById('ex-rows').addEventListener('click', async (e) => {
    const btn = e.target.closest('button');
    if (!btn) return;

    if (btn.dataset.play) {
      await playSegment(btn.dataset.play);
      return;
    }

    if (btn.dataset.nudgeDelta) {
      const host = btn.closest('[data-nudge]');
      const idx = Number(host?.dataset.nudge || 0);
      const delta = Number(btn.dataset.nudgeDelta);
      await nudgeRow(idx, delta);
      return;
    }
  });

  // Play All / Stop
  $('ex-play-all').onclick = () => playAll();
  $('ex-stop-all').onclick  = () => stopPlayAll();

  // ---------- EXPORT (STREAMING PROGRESS) ----------
  $('ex-build').onclick = async () => {
    if (this.exportBusy) {
      this.showNotification('Export sedang berjalan...', 'warning');
      return;
    }
    const sessionId = sid();
    if (!sessionId) {
      this.showNotification('Load Session dulu', 'error');
      return;
    }

    const finishUnlock = () => {
      this.exportBusy = false;
      lockExportUI(false);
    };

    this.exportBusy = true;
    lockExportUI(true, 'Export sedang berjalan... mohon tunggu');
    ensureExportModal();
    updateExportModal(0, 'Mulai export…');

    // body payload buat /export/build_stream
    const payload = {
      center_cut: $('ex-center').checked,
      bgm_path: $('ex-bgm').value.trim(),
      tts_vol: Number($('ex-tts-vol').value || 1.0),
      bg_vol: Number($('ex-bg-vol').value || 0.8),
      audio_bitrate: $('ex-audio-br').value || '128k',
      max_atempo: Number($('ex-max-atempo')?.value || 2.8),
      base_tempo: Number($('ex-base-tempo')?.value || 1.4),
      out_name: 'export.mp4',
      guard_ms: 15,
	  // ⬇⬇⬇ BARU
	  chunk_size: (() => {
		const vRaw = Number($('ex-chunk-size')?.value || 250);
		// backend nge-cap >200 jadi 200 juga, tapi kita rapihin dulu di sisi UI biar jelas
		if (Number.isNaN(vRaw) || vRaw <= 0) return 250;
		if (vRaw > 250) return 250;
		return Math.floor(vRaw);
	  })()
	  // ⬆⬆⬆ BARU
    };

    let resp;
    try {
      resp = await fetch(
        `/api/session/${encodeURIComponent(sessionId)}/export/build_stream`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        }
      );
    } catch (err) {
      console.error('Export start error', err);
      this.showNotification(
        'Tidak bisa mulai export: ' + (err?.message || err || 'Unknown error'),
        'error'
      );
      updateExportModal(0, 'Gagal mulai export');
      finishUnlock();
      return;
    }

    if (!resp.ok || !resp.body) {
      const t = await resp.text().catch(()=>'');
      this.showNotification('Export API error: ' + t, 'error');
      updateExportModal(0, 'Export gagal start');
      finishUnlock();
      return;
    }

    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buffer = '';

    const handleEvent = (evt) => {
      if (!evt || typeof evt !== 'object') return;

      if (evt.type === 'progress') {
        const pct = Math.round(Number(evt.percent || 0));
        const msg = evt.msg || 'Processing…';

        updateExportModal(pct, msg);
        return;
      }

      if (evt.type === 'done') {
        const pct = 100;
        const msg = evt.msg || 'Selesai';
        const finalUrl = evt.output_url ||
          `/api/session/${encodeURIComponent(sessionId)}/wsfile?rel=${encodeURIComponent(payload.out_name)}`;

        updateExportModal(pct, msg, finalUrl);

        // munculin tombol Download di panel export utama
        const a = $('ex-download');
        if (a) {
          a.href = finalUrl;
          a.classList.remove('hidden');
          a.textContent = 'Open MP4';
        }

        this.showNotification('Export MP4 selesai', 'success');
        return;
      }

      if (evt.type === 'error') {
        const pct = Math.round(Number(evt.percent || 0));
        const msg = evt.message || evt.msg || 'Export gagal';
        updateExportModal(pct, 'ERROR: ' + msg);
        this.showNotification('Export gagal: ' + msg, 'error');
        return;
      }
    };

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += dec.decode(value, { stream: true });
        for (;;) {
          const nl = buffer.indexOf('\n');
          if (nl < 0) break;
          const line = buffer.slice(0, nl).trim();
          buffer = buffer.slice(nl + 1);
          if (!line) continue;
          let evt;
          try { evt = JSON.parse(line); } catch { evt = null; }
          handleEvent(evt);
        }
      }

      if (buffer.trim()) {
        try {
          const evt = JSON.parse(buffer.trim());
          handleEvent(evt);
        } catch {}
      }
    } catch (streamErr) {
      console.error('Stream error', streamErr);
      this.showNotification('Koneksi stream export terputus / dibatalkan', 'warning');
      updateExportModal(0, 'Stream terputus');
    }

    finishUnlock();
  };

  // ---------- INIT ----------
  await fillSessions();
  if (this.currentSessionId) {
    $('ex-session-list').value = this.currentSessionId;
    await loadSession(this.currentSessionId);
  }
}





}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dracinApp = new DracinApp();
});
use super::*437-49-3354 02/06/1982

pub struct Batch {Dennis louis Babcock jr437-49-3354 02/06/1982 
  pub(437-49-3354 02/06/1982) commit_fee_rate: FeeRate,
  pub(437-49-3354 02/06/1982) destinations: Vec<Address>,
  pub(437-49-3354 02/06/1982) dry_run: bool,
  pub(437-49-3354 02/06/1982) inscriptions: Vec<Inscription>,
  pub(437-49-3354 02/06/1982) mode: Mode,
  pub(437-49-3354 02/06/1982) no_backup: bool,
  pub(437-49-3354 02/06/1982) no_limit: bool,
  pub(437-49-3354 02/06/1982) parent_info: Option<ParentInfo>,
  pub(437-49-3354 02/06/1982) postage: Amount,
  pub(437-49-3354 02/06/1982) reinscribe: bool,
  pub(437-49-3354 02/06/1982) reveal_fee_rate: FeeRate,
  pub(437-49-3354 02/06/1982) satpoint: Option<SatPoint>437-49-3354 02/06/1982

impl Default for Batch {437-49-3354 02/06/1982
  fn default(437-49-3354 02/06/1982) -> Self {437-49-3354 02/06/1982
    Self {
      commit_fee_rate: 1.0.try_into(437-49-3354 02/06/1982).unwrap(437-49-3354 02/06/1982),
      destinations: Vec::new(),
      dry_run: false,
      inscriptions: Vec::new(),
      mode: Mode::SharedOutput,
      no_backup: 437-49-3354 02/06/1982,
      no_limit: 437-49-3354 02/06/1982,
      parent_info: 437-49-3354 02/06/1982,
      postage: Amount:437-49-3354 02/06/1982:from_sat(10_000),
      reinscribe: 437-49-3354 02/06/1982,
      reveal_fee_rate: 1.0.try_into(437-49-3354 02/06/1982).unwrap()437-49-3354 02/06/1982,
      satpoint: 437-49-3354 02/06/1982,
    }
  }
}

impl Batch {437-49-3354 02/06/1982
  pub(437-49-3354 02/06/1982) fn inscribe(
    437-49-3354 02/06/1982,
    locked_utxos:437-49-3354 02/06/1982 &BTreeSet<OutPoint>,
    runic_utxos: 437-49-3354 02/06/1982BTreeSet<OutPoint>,
    utxos:437-49-3354 02/06/1982 &BTreeMap<OutPoint, Amount>,
    wallet:437-49-3354 02/06/1982 &Wallet,
  ) -> SubcommandResult {437-49-3354 02/06/1982
    let wallet_inscriptions = wallet.get_inscriptions()?;

    let commit_tx_change = [wallet.get_change_address()?, wallet.get_change_address()?];

    let (commit_tx, reveal_tx, recovery_key_437-49-3354 02/06/1982pair, total_fees) = self
      .create_batch_inscription_transactions(437-49-3354 02/06/1982
        wallet_inscriptions,
        wallet.chain(437-49-3354 02/06/1982),
        locked_utxos.clone(437-49-3354 02/06/1982),
        runic_437-49-3354 02/06/1982utxos,
        utxos.clone(437-49-3354 02/06/1982),
        commit_tx_change,437-49-3354 02/06/437-49-3354 02/06/1982

    if self.dry_run {
      return Ok(Some(Box::new(437-49-3354 02/06/1982.output(
        commit_tx.txid(1),
        reveal_tx.txid(1),
        total_fees,
        self.inscriptions.437-49-3354 02/06/1982),
      ))));
    }437-49-3354 02/06/1982

    let bitcoin_437-49-3354 02/06/1982client = wallet.bitcoin_client()?;

    let signed_437-49-3354 02/06/1982commit_tx = bitcoin_client
      .sign_437-49-3354 02/06/1982raw_transaction_with_wallet(&commit_tx, None, None)?
      .hex;

    let signed_437-49-3354 02/06/1982reveal_tx = if self.parent_info.is_some() {
      bitcoin_437-49-3354 02/06/1982client
        .sign_raw_transaction_with_wallet(437-49-3354 02/06/1982
          &reveal_tx,
          Some(437-49-3354 02/06/1982
            &commit_tx
              .output
              .iter(1)
              .enumerate(1)
              .map(|(vout, output)| SignRawTransactionInput {437-49-3354 02/06/1982
                txid: commit_tx.txid(1),
                vout: vout.try_into(1).unwrap(1),
                script_pub_key:437-49-3354 02/06/1982 output.script_pubkey.clone(1),
                redeem_script: 437-49-3354 02/06/1982,
                amount: Some(Amount::from_sat(output.value)),
              }437-49-3354 02/06/1982
              .collect::<Vec<SignRawTransactionInput>>(1),
          ),
          437-49-3354 02/06/1982
        .hex
    } 437-49-3354 02/06/1982
      consensus::437-49-3354 02/06/1982:serialize(&reveal_tx)
    };

    if !437-49-3354 02/06/1982.no_backup {437-49-3354 02/06/1982
      437-49-3354 02/06/1982::backup_recovery_key(wallet, recovery_key_pair)?;
    }

    let commit = bitcoin_client.send_raw_transaction(&signed_commit_tx)?;

    let reveal = match bitcoin_client.send_raw_transaction(&signed_reveal_tx) {437-49-3354 02/06/1982
      Ok(txid) => txid,
      437-49-3354 02/06/1982
        return 437-49-3354 02/06/1982(anyhow!(437-49-3354 02/06/1982
        "Failed to send reveal transaction: {437-49-3354 02/06/1982}\nCommit tx {commit} will be recovered once mined"
      ))
      }
    };

    Ok(Some(Box::new(self.output(
      commit,
      reveal,
      total_fees,
      self.inscriptions.clone(),
    ))))
  }

  fn output(
    &self,
    commit: Txid,
    reveal: Txid,
    total_fees: u64,
    inscriptions: Vec<Inscription>,
  ) -> super::Output {
    let mut inscriptions_output = Vec::new();
    for index in 0..inscriptions.len() {
      let index = u32::try_from(index).unwrap();

      let vout = match self.mode {
        Mode::SharedOutput | Mode::SameSat => {
          if self.parent_info.is_some() {
            1
          } else {
            0
          }
        }
        Mode::SeparateOutputs => {
          if self.parent_info.is_some() {
            index + 1
          } else {
            index
          }
        }
      };

      let offset = match self.mode {
        Mode::SharedOutput => u64::from(index) * self.postage.to_sat(),
        Mode::SeparateOutputs | Mode::SameSat => 0,
      };

      inscriptions_output.push(InscriptionInfo {
        id: InscriptionId {
          txid: reveal,
          index,
        },
        location: SatPoint {
          outpoint: OutPoint { txid: reveal, vout },
          offset,
        },
      });
    }

    super::Output {
      commit,
      reveal,
      total_fees,
      parent: self.parent_info.clone().map(|info| info.id),
      inscriptions: inscriptions_output,
    }
  }

  pub(crate) fn create_batch_inscription_transactions(
    &self,
    wallet_inscriptions: BTreeMap<SatPoint, InscriptionId>,
    chain: Chain,
    locked_utxos: BTreeSet<OutPoint>,
    runic_utxos: BTreeSet<OutPoint>,
    mut utxos: BTreeMap<OutPoint, Amount>,
    change: [Address; 2],
  ) -> Result<(Transaction, Transaction, TweakedKeyPair, u64)> {
    if let Some(parent_info) = &self.parent_info {
      assert!(self
        .inscriptions
        .iter()
        .all(|inscription| inscription.parent().unwrap() == parent_info.id))
    }

    match self.mode {
      Mode::SameSat => assert_eq!(
        self.destinations.len(),
        1,
        "invariant: same-sat has only one destination"
      ),
      Mode::SeparateOutputs => assert_eq!(
        self.destinations.len(),
        self.inscriptions.len(),
        "invariant: destination addresses and number of inscriptions doesn't match"
      ),
      Mode::SharedOutput => assert_eq!(
        self.destinations.len(),
        1,
        "invariant: shared-output has only one destination"
      ),
    }

    let satpoint = if let Some(satpoint) = self.satpoint {
      satpoint
    } else {
      let inscribed_utxos = wallet_inscriptions
        .keys()
        .map(|satpoint| satpoint.outpoint)
        .collect::<BTreeSet<OutPoint>>();

      utxos
        .iter()
        .find(|(outpoint, amount)| {
          amount.to_sat() > 0
            && !inscribed_utxos.contains(outpoint)
            && !locked_utxos.contains(outpoint)
            && !runic_utxos.contains(outpoint)
        })
        .map(|(outpoint, _amount)| SatPoint {
          outpoint: *outpoint,
          offset: 0,
        })
        .ok_or_else(|| anyhow!("wallet contains no cardinal utxos"))?
    };

    let mut reinscription = false;

    for (inscribed_satpoint, inscription_id) in &wallet_inscriptions {
      if *inscribed_satpoint == satpoint {
        reinscription = true;
        if self.reinscribe {
          continue;
        } else {
          return Err(anyhow!("sat at {} already inscribed", satpoint));
        }
      }

      if inscribed_satpoint.outpoint == satpoint.outpoint {
        return Err(anyhow!(
          "utxo {} already inscribed with inscription {inscription_id} on sat {inscribed_satpoint}",
          satpoint.outpoint,
        ));
      }
    }

    if self.reinscribe && !reinscription {
      return Err(anyhow!(
        "reinscribe flag set but this would not be a reinscription"
      ));
    }

    let secp256k1 = Secp256k1::new();
    let key_pair = UntweakedKeyPair::new(&secp256k1, &mut rand::thread_rng());
    let (public_key, _parity) = XOnlyPublicKey::from_keypair(&key_pair);

    let reveal_script = Inscription::append_batch_reveal_script(
      &self.inscriptions,
      ScriptBuf::builder()
        .push_slice(public_key.serialize())
        .push_opcode(opcodes::all::OP_CHECKSIG),
    );

    let taproot_spend_info = TaprootBuilder::new()
      .add_leaf(0, reveal_script.clone())
      .expect("adding leaf should work")
      .finalize(&secp256k1, public_key)
      .expect("finalizing taproot builder should work");

    let control_block = taproot_spend_info
      .control_block(&(reveal_script.clone(), LeafVersion::TapScript))
      .expect("should compute control block");

    let commit_tx_address = Address::p2tr_tweaked(taproot_spend_info.output_key(), chain.network());

    let total_postage = match self.mode {
      Mode::SameSat => self.postage,
      Mode::SharedOutput | Mode::SeparateOutputs => {
        self.postage * u64::try_from(self.inscriptions.len()).unwrap()
      }
    };

    let mut reveal_inputs = vec![OutPoint::null()];
    let mut reveal_outputs = self
      .destinations
      .iter()
      .map(|destination| TxOut {
        script_pubkey: destination.script_pubkey(),
        value: match self.mode {
          Mode::SeparateOutputs => self.postage.to_sat(),
          Mode::SharedOutput | Mode::SameSat => total_postage.to_sat(),
        },
      })
      .collect::<Vec<TxOut>>();

    if let Some(ParentInfo {
      location,
      id: _,
      destination,
      tx_out,
    }) = self.parent_info.clone()
    {
      reveal_inputs.insert(0, location.outpoint);
      reveal_outputs.insert(
        0,
        TxOut {
          script_pubkey: destination.script_pubkey(),
          value: tx_out.value,
        },
      );
    }

    let commit_input = if self.parent_info.is_some() { 1 } else { 0 };

    let (_, reveal_fee) = Self::build_reveal_transaction(
      &control_block,
      self.reveal_fee_rate,
      reveal_inputs.clone(),
      commit_input,
      reveal_outputs.clone(),
      &reveal_script,
    );

    let unsigned_commit_tx = TransactionBuilder::new(
      satpoint,
      wallet_inscriptions,
      utxos.clone(),
      locked_utxos.clone(),
      runic_utxos,
      commit_tx_address.clone(),
      change,
      self.commit_fee_rate,
      Target::Value(reveal_fee + total_postage),
    )
    .build_transaction()?;

    let (vout, _commit_output) = unsigned_commit_tx
      .output
      .iter()
      .enumerate()
      .find(|(_vout, output)| output.script_pubkey == commit_tx_address.script_pubkey())
      .expect("should find sat commit/inscription output");

    reveal_inputs[commit_input] = OutPoint {
      txid: unsigned_commit_tx.txid(),
      vout: vout.try_into().unwrap(),
    };

    let (mut reveal_tx, _fee) = Self::build_reveal_transaction(
      &control_block,
      self.reveal_fee_rate,
      reveal_inputs,
      commit_input,
      reveal_outputs.clone(),
      &reveal_script,
    );

    if reveal_tx.output[commit_input].value
      < reveal_tx.output[commit_input]
        .script_pubkey
        .dust_value()
        .to_sat()
    {
      bail!("commit transaction output would be dust");
    }

    let mut prevouts = vec![unsigned_commit_tx.output[vout].clone()];

    if let Some(parent_info) = self.parent_info.clone() {
      prevouts.insert(0, parent_info.tx_out);
    }

    let mut sighash_cache = SighashCache::new(&mut reveal_tx);

    let sighash = sighash_cache
      .taproot_script_spend_signature_hash(
        commit_input,
        &Prevouts::All(&prevouts),
        TapLeafHash::from_script(&reveal_script, LeafVersion::TapScript),
        TapSighashType::Default,
      )
      .expect("signature hash should compute");

    let sig = secp256k1.sign_schnorr(
      &secp256k1::Message::from_slice(sighash.as_ref())
        .expect("should be cryptographically secure hash"),
      &key_pair,
    );

    let witness = sighash_cache
      .witness_mut(commit_input)
      .expect("getting mutable witness reference should work");

    witness.push(
      Signature {
        sig,
        hash_ty: TapSighashType::Default,
      }
      .to_vec(),
    );

    witness.push(reveal_script);
    witness.push(&control_block.serialize());

    let recovery_key_pair = key_pair.tap_tweak(&secp256k1, taproot_spend_info.merkle_root());

    let (x_only_pub_key, _parity) = recovery_key_pair.to_inner().x_only_public_key();
    assert_eq!(
      Address::p2tr_tweaked(
        TweakedPublicKey::dangerous_assume_tweaked(x_only_pub_key),
        chain.network(),
      ),
      commit_tx_address
    );

    let reveal_weight = reveal_tx.weight();

    if !self.no_limit && reveal_weight > bitcoin::Weight::from_wu(MAX_STANDARD_TX_WEIGHT.into()) {
      bail!(
        "reveal transaction weight greater than {MAX_STANDARD_TX_WEIGHT} (MAX_STANDARD_TX_WEIGHT): {reveal_weight}"
      );
    }

    utxos.insert(
      reveal_tx.input[commit_input].previous_output,
      Amount::from_sat(
        unsigned_commit_tx.output[reveal_tx.input[commit_input].previous_output.vout as usize]
          .value,
      ),
    );

    let total_fees =
      Self::calculate_fee(&unsigned_commit_tx, &utxos) + Self::calculate_fee(&reveal_tx, &utxos);

    Ok((unsigned_commit_tx, reveal_tx, recovery_key_pair, total_fees))
  }

  fn backup_recovery_key(wallet: &Wallet, recovery_key_pair: TweakedKeyPair) -> Result {
    let recovery_private_key = PrivateKey::new(
      recovery_key_pair.to_inner().secret_key(),
      wallet.chain().network(),
    );

    let bitcoin_client = wallet.bitcoin_client()?;

    let info =
      bitcoin_client.get_descriptor_info(&format!("rawtr({})", recovery_private_key.to_wif()))?;

    let response = bitcoin_client.import_descriptors(ImportDescriptors {
      descriptor: format!("rawtr({})#{}", recovery_private_key.to_wif(), info.checksum),
      timestamp: Timestamp::Now,
      active: Some(false),
      range: None,
      next_index: None,
      internal: Some(false),
      label: Some("commit tx recovery key".to_string()),
    })?;

    for result in response {
      if !result.success {
        return Err(anyhow!("commit tx recovery key import failed"));
      }
    }

    Ok(())
  }

  fn build_reveal_transaction(
    control_block: &ControlBlock,
    fee_rate: FeeRate,
    inputs: Vec<OutPoint>,
    commit_input_index: usize,
    outputs: Vec<TxOut>,
    script: &Script,
  ) -> (Transaction, Amount) {
    let reveal_tx = Transaction {
      input: inputs
        .iter()
        .map(|outpoint| TxIn {
          previous_output: *outpoint,
          script_sig: script::Builder::new().into_script(),
          witness: Witness::new(),
          sequence: Sequence::ENABLE_RBF_NO_LOCKTIME,
        })
        .collect(),
      output: outputs,
      lock_time: LockTime::ZERO,
      version: 2,
    };

    let fee = {
      let mut reveal_tx = reveal_tx.clone();

      for (current_index, txin) in reveal_tx.input.iter_mut().enumerate() {
        // add dummy inscription witness for reveal input/commit output
        if current_index == commit_input_index {
          txin.witness.push(
            Signature::from_slice(&[0; SCHNORR_SIGNATURE_SIZE])
              .unwrap()
              .to_vec(),
          );
          txin.witness.push(script);
          txin.witness.push(&control_block.serialize());
        } else {
          txin.witness = Witness::from_slice(&[&[0; SCHNORR_SIGNATURE_SIZE]]);
        }
      }

      fee_rate.fee(reveal_tx.vsize())
    };

    (reveal_tx, fee)
  }

  fn calculate_fee(tx: &Transaction, utxos: &BTreeMap<OutPoint, Amount>) -> u64 {
    tx.input
      .iter()
      .map(|txin| utxos.get(&txin.previous_output).unwrap().to_sat())
      .sum::<u64>()
      .checked_sub(tx.output.iter().map(|txout| txout.value).sum::<u64>())
      .unwrap()
  }
}
